from __future__ import annotations

import numpy as np
from movie_revenue_prediction.utils.functions import eval_on_val, chronosort_for_tscv
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks


# __________________ MLP _______________
def build_mlp(
    n_features: int,
    hidden_layers: int = 1,
    hidden_units: int = 128,
    dropout: float = 0.1,
    l2_reg: float = 5e-4,
    batch_norm: bool = False,
    activation: str = "swish",   # keep default
    learning_rate: float = 3e-4,
    loss_name: str = "huber",
    wide: bool = True,
    clipnorm: float | None = 1.0,
    residual: bool = False
):
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for _ in range(hidden_layers):
        h = layers.Dense(
            hidden_units,
            activation=activation,   
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        if batch_norm:
            h = layers.BatchNormalization()(h)
        if dropout and dropout > 0:
            h = layers.Dropout(dropout)(h)
        x = layers.Add()([x, h]) if (residual and x.shape[-1] == h.shape[-1]) else h

    deep_out = layers.Dense(1, activation="linear")(x)
    if wide:
        wide_out = layers.Dense(1, activation="linear",
                                use_bias=True,
                                kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        outputs = layers.Add()([deep_out, wide_out])
    else:
        outputs = deep_out

    model = keras.Model(inputs, outputs)

    loss = (keras.losses.Huber(delta=1.0) if loss_name == "huber"
            else keras.losses.LogCosh() if loss_name == "logcosh"
            else "mse")
    opt = keras.optimizers.Adam(learning_rate=learning_rate,
                               clipnorm=clipnorm if clipnorm else None)


    model.compile(optimizer=opt, loss=loss,
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae"),
                           keras.metrics.RootMeanSquaredError(name="rmse")])
    return model





def fit_nn_cv(
    X, y,
    n_iter: int = 15,
    n_splits: int = 5,
    verbose: int = 1,
    step_name: str = "nn"
):
    """
    Impute -> Scale -> KerasRegressor (MLP), with TimeSeriesSplit + RandomizedSearchCV.
    Uses RMSE scorer (lower is better). EarlyStopping is enabled during CV (on training loss).
    Returns the fitted search object.
    """
    # Chronological split like your other models
    X_sorted, y_sorted = chronosort_for_tscv(X, y, year_col="x_year")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # KerasRegressor with model builder
    n_features = X_sorted.shape[1]
    reg = KerasRegressor(
        model=build_mlp,
        model__n_features=n_features,
        epochs=120,          # CV ceiling; EarlyStopping will stop earlier
        batch_size=64,
        verbose=0
    )

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (step_name, reg)
    ])

    # Search space (compact & stable)
    param_dist = {
        f"{step_name}__model__hidden_layers": [1, 2, 3],
        f"{step_name}__model__hidden_units": [128, 256],
        f"{step_name}__model__dropout": [0.0, 0.05, 0.1],
        f"{step_name}__model__l2_reg": [1e-5, 1e-4, 1e-3], 
        f"{step_name}__model__batch_norm": [False, True],
        f"{step_name}__model__learning_rate": [3e-4, 5e-4],
        f"{step_name}__model__loss_name": ["huber", "mse"],
        f"{step_name}__model__wide": [True, False],
        f"{step_name}__model__activation": ["relu"], 
        f"{step_name}__model__clipnorm": [1.0],
        f"{step_name}__batch_size": [32],
        f"{step_name}__model__residual": [False],
        f"{step_name}__epochs": [20, 30],
    }



    rmse_scorer = make_scorer(
        lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        greater_is_better=False
    )

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=rmse_scorer,
        cv=tscv,
        n_jobs=1,               # keep 1 for TF stability
        random_state=42,
        verbose=verbose,
        refit=True
        # set error_score="raise" temporarily if you want to debug failing configs
    )

    # EarlyStopping DURING CV (train loss, since CV folds don't pass validation_data)
    es_cv = callbacks.EarlyStopping(
        monitor="loss", mode="min", patience=5,
        restore_best_weights=False, verbose=0
    )

    # Important: do NOT pass 'workers'/'use_multiprocessing' with Keras 3
    search.fit(
        X_sorted, y_sorted,
        **{
            f"{step_name}__callbacks": [es_cv],
            f"{step_name}__verbose": 0
        }
    )
    return search


class SWA(keras.callbacks.Callback):
    """
    Stochastic Weight Averaging: starting at `start_epoch`, keep a running
    average of model weights every `update_freq` epochs. At training end,
    set the model weights to the averaged weights.

    Notes:
    - Put this callback LAST in the callbacks list so it runs after EarlyStopping.
    - Works with any optimizer (Adam/SGD). Original papers use SGD + cyclical LR,
      but averaging with Adam still gives a small, reliable gain.
    """
    def __init__(self, start_epoch: int = 10, update_freq: int = 1):
        super().__init__()
        self.start_epoch = int(start_epoch)
        self.update_freq = int(update_freq)
        self.swa_weights = None
        self.n_averaged = 0

    def on_epoch_end(self, epoch, logs=None):
        epoch_idx = epoch + 1  # make human-friendly
        if epoch_idx >= self.start_epoch and ((epoch_idx - self.start_epoch) % self.update_freq == 0):
            weights = self.model.get_weights()
            if self.swa_weights is None:
                # initialize running average
                self.swa_weights = [w.astype("float32").copy() for w in weights]
                self.n_averaged = 1
            else:
                self.n_averaged += 1
                beta = 1.0 / self.n_averaged
                for i in range(len(self.swa_weights)):
                    # SWA running average: w_avg = (1-β)*w_avg + β*w
                    self.swa_weights[i] = (1.0 - beta) * self.swa_weights[i] + beta * weights[i]

    def on_train_end(self, logs=None):
        if self.swa_weights is not None and self.n_averaged > 0:
            self.model.set_weights(self.swa_weights)

def build_callbacks_for_final_with_swa(
    earlystop_patience=4, reducelr_patience=2, min_delta=5e-4,
    swa_start_epoch: int = 10, swa_update_freq: int = 1, use_checkpoint: bool = True
):
    es = callbacks.EarlyStopping(
        monitor="val_loss", mode="min",
        patience=earlystop_patience, min_delta=min_delta,
        restore_best_weights=True
    )
    rlrop = callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min",
        factor=0.5, patience=reducelr_patience, min_lr=1e-6, verbose=0
    )
    cbs = [es, rlrop]
    if use_checkpoint:
        ckpt = callbacks.ModelCheckpoint(
            filepath="artifacts/nn_best_weights.keras",
            monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=False, verbose=0
        )
        cbs.append(ckpt)
    # IMPORTANT: SWA LAST so it runs after EarlyStopping stops training
    cbs.append(SWA(start_epoch=swa_start_epoch, update_freq=swa_update_freq))
    return cbs


def run_nn_experiment_with_swa(
    X_train, y_train, X_val, y_val,
    version_name: str = "NN (MLP) + SWA",
    step_name: str = "nn",
    earlystop_patience: int = 4,
    reducelr_patience: int = 2,
    n_iter: int = 16,
    n_splits: int = 5,
    verbose: int = 1,
    final_max_epochs: int = 35,
    earlystop_min_delta: float = 5e-4,
    swa_start_epoch: int = 10,
    swa_update_freq: int = 1
):
    print(f"\n Running {version_name} ...")
    # 1) do the usual time-aware CV search (uses your existing fit_nn_cv)
    search = fit_nn_cv(X_train, y_train, n_iter=n_iter, n_splits=n_splits, verbose=verbose, step_name=step_name)
    best_pipe = search.best_estimator_

    # 2) final refit WITH SWA + internal validation (post-preprocessing)
    cbs = build_callbacks_for_final_with_swa(
        earlystop_patience=earlystop_patience,
        reducelr_patience=reducelr_patience,
        min_delta=earlystop_min_delta,
        swa_start_epoch=swa_start_epoch,
        swa_update_freq=swa_update_freq,
        use_checkpoint=True
    )
    fit_params = {
        f"{step_name}__callbacks": cbs,
        f"{step_name}__validation_split": 0.15,
        f"{step_name}__verbose": 0,
        f"{step_name}__epochs": final_max_epochs,
    }
    best_pipe.fit(X_train, y_train, **fit_params)

    # Optional: load the checkpointed best (EarlyStopping) — SWA already set weights at end.
    try:
        nn_step = best_pipe.named_steps[step_name]
        best_model = keras.models.load_model("artifacts/nn_best_weights.keras")
        # If you prefer the SWA weights instead of the checkpoint, comment the next line.
        # Here we keep SWA final weights (already set), but loading checkpoint is available if you want to compare.
        # nn_step.model_ = best_model
    except Exception:
        pass

    # History (SWA doesn't change how we collect it)
    nn_step = best_pipe.named_steps[step_name]
    history = getattr(nn_step, "history_", None)
    history = {k: list(v) for k, v in history.items()} if isinstance(history, dict) else {}

    metrics = eval_on_val(best_pipe, X_val, y_val)
    print(f"Best Params (search): {search.best_params_}")
    print(f"Validation → RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R²: {metrics['R2']:.4f}")
    return best_pipe, search.best_params_, metrics, search, history




def _nn_final_callbacks(es_pat=4, rlrop_pat=2, min_delta=5e-4, swa_start=10):
    es = callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                 patience=es_pat, min_delta=min_delta,
                                 restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                     factor=0.5, patience=rlrop_pat,
                                     min_lr=1e-6, verbose=0)
    return [es, rl, SWA(start_epoch=swa_start, update_freq=1)]

def make_nn_from_saved(saved: dict, n_features: int, step_name="nn") -> Pipeline:
    # Build KerasRegressor and set saved params (keys start with 'nn__...')
    reg = KerasRegressor(
        model=build_mlp,
        model__n_features=n_features,
        # sensible defaults; will be overridden by saved best_params
        epochs=40, batch_size=32, verbose=0
    )
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (step_name, reg)
    ])
    # Map saved keys directly (they include 'nn__...')
    pipe.set_params(**saved["best_params"])
    return pipe