"""
TMDB enrichment and feature engineering utilities for the movie revenue project.

Two main responsibilities:

1. TMDB enrichment (formerly `features2.py`)
   - TMDBClient + helpers to call the TMDB API.
   - build_core_movie_frame / run_core_enrichment_pipeline to construct a
     clean, enriched movie DataFrame.

2. Feature engineering (formerly `features.py`)
   - Helper utilities to build one-hot / multi-hot encodings.
   - make_features_timeaware_splits to turn an enriched movie frame into
     modeling-ready train/val/test feature matrices.

"""

# Imports
from __future__ import annotations
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import requests

# TMDB enrichment configuration (from former features2.py)
DEFAULT_TOP_CAST_K = 3
DEFAULT_WORKERS = 12
DEFAULT_DROP_THRESHOLDS = (1_000, 1_000)
DEFAULT_PRIORITY_COUNTRIES = ("US", "GB", "DK", "CA", "AU", "IN")
TMDB_BASE = "https://api.themoviedb.org/3"

# TMDB enrichment client and helpers
@dataclass
class TMDBClient:
    api_key: str
    session: requests.Session

    @classmethod
    def from_env(cls) -> "TMDBClient":
        key = os.getenv("TMDB_KEY")
        if not key:
            raise RuntimeError("Please set TMDB_KEY environment variable.")
        session = requests.Session()
        session.params = {"api_key": key}
        session.headers.update({"Accept": "application/json"})
        return cls(api_key=key, session=session)

    def get_movie(self, movie_id: int, append_to_response: str = "") -> Dict[str, Any]:
        params = {"append_to_response": append_to_response} if append_to_response else {}
        r = self.session.get(
            f"{TMDB_BASE}/movie/{int(movie_id)}",
            params=params,
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    def movie_details(self, movie_id: int) -> Dict[str, Any]:
        r = self.session.get(f"{TMDB_BASE}/movie/{int(movie_id)}", timeout=20)
        r.raise_for_status()
        return r.json()

    def movie_credits(self, movie_id: int) -> Dict[str, Any]:
        r = self.session.get(f"{TMDB_BASE}/movie/{int(movie_id)}/credits", timeout=20)
        r.raise_for_status()
        return r.json()

def _none_if_empty(x):
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    if isinstance(x, (list, tuple, set, dict)):
        return x if len(x) else None
    return x

def _list_of_names(people) -> Optional[List[str]]:
    if not people:
        return None
    out = []
    for p in people:
        name = _none_if_empty(p.get("name"))
        if name:
            out.append(name)
    return out or None

def _map_gender(g) -> str:
    if g is None or g == 0:
        return "unknown"
    if g == 1:
        return "female"
    if g == 2:
        return "male"
    if g == 3:
        return "non-binary"
    return "unknown"

def _extract_certification(
    release_dates_block: Dict[str, Any],
    priority_countries: Sequence[str] = DEFAULT_PRIORITY_COUNTRIES,
) -> Optional[str]:
    if not release_dates_block:
        return None

    results = release_dates_block.get("results") or []
    if not results:
        return None

    by_country = {r.get("iso_3166_1"): (r.get("release_dates") or []) for r in results}
    buckets = list(priority_countries) + [None]

    for country in buckets:
        for r in by_country.get(country, []):
            cert = _none_if_empty(r.get("certification"))
            if cert:
                return cert
    return None

def _extract_composers_from_crew(crew, max_n: int = 5):
    if not crew:
        return None, None

    def _norm(s):
        return (s or "").strip().lower()

    p1_jobs = {
        "original music composer", "music by", "main title theme composer",
        "original score composer", "score composer"
    }
    p2_jobs = {"composer"}
    p3_jobs = {
        "theme composer", "additional music", "score producer", "musical score producer",
        "co-composer"
    }

    exclude_jobs = {
        "music supervisor", "music editor", "orchestrator", "orchestration",
        "conductor", "score mixer", "music mixer", "arranger",
        "music coordinator", "music consultant", "song", "songs",
        "songwriter", "lyrics", "lyricist", "sound", "sound designer",
        "sound editor", "sound effects editor", "music recordist",
        "music preparation", "music preparation services"
    }

    buckets = {1: [], 2: [], 3: []}
    for member in crew:
        job = _norm(member.get("job"))
        name = _none_if_empty(member.get("name"))
        if not name or not job:
            continue
        if any(ex in job for ex in exclude_jobs):
            continue

        if job in p1_jobs:
            buckets[1].append(name)
        elif job in p2_jobs:
            buckets[2].append(name)
        elif job in p3_jobs:
            buckets[3].append(name)

    for level in (1, 2, 3):
        if buckets[level]:
            uniq = list(dict.fromkeys(buckets[level]))
            return uniq[:max_n], len(uniq)

    return None, None

def _fetch_movie_core(
    mid: int,
    client: TMDBClient,
    *,
    top_cast_k: int = DEFAULT_TOP_CAST_K,
    priority_countries: Sequence[str] = DEFAULT_PRIORITY_COUNTRIES,
) -> Optional[Dict[str, Any]]:
    try:
        data = client.get_movie(
            mid,
            append_to_response="release_dates,keywords,credits",
        )
    except Exception:
        time.sleep(0.5)
        try:
            data = client.movie_details(mid)
            credits = client.movie_credits(mid)
            data["credits"] = credits
        except Exception:
            return None

    if not data:
        return None

    release_dates_block = (data.get("release_dates") or {}).get("results") or []
    release_dates_norm = {"results": release_dates_block}

    cert = _extract_certification(release_dates_norm, priority_countries)

    crew = ((data.get("credits") or {}).get("crew")) or []
    cast = ((data.get("credits") or {}).get("cast")) or []

    directors = [m.get("name") for m in crew if (m.get("job") or "").lower() == "director"]
    directors = [d for d in directors if _none_if_empty(d)]
    directors = directors or None

    lead_cast = []
    lead_cast_genders = []
    cast_sorted = sorted(cast, key=lambda x: (x.get("order") is None, x.get("order")), reverse=False)
    for m in cast_sorted[:top_cast_k]:
        name = _none_if_empty(m.get("name"))
        if not name:
            continue
        lead_cast.append(name)
        lead_cast_genders.append(_map_gender(m.get("gender")))

    lead_cast = lead_cast or None
    lead_cast_genders = lead_cast_genders or None

    status = _none_if_empty(data.get("status"))
    original_language = _none_if_empty(data.get("original_language"))

    runtime = data.get("runtime")
    revenue = data.get("revenue")
    budget = data.get("budget")
    popularity = data.get("popularity")
    vote_count = data.get("vote_count")
    vote_average = data.get("vote_average")
    homepage = _none_if_empty(data.get("homepage"))

    collection_name = _none_if_empty((data.get("belongs_to_collection") or {}).get("name"))

    genres = _list_of_names(data.get("genres") or [])
    production_countries = [
        _none_if_empty(d.get("name"))
        for d in (data.get("production_countries") or [])
        if _none_if_empty(d.get("name"))
    ] or None

    spoken_languages = [
        _none_if_empty(d.get("english_name") or d.get("name"))
        for d in (data.get("spoken_languages") or [])
        if _none_if_empty(d.get("english_name") or d.get("name"))
    ] or None
    num_spoken_languages = len(spoken_languages) if spoken_languages else None

    kw_block = data.get("keywords") or {}
    kw_list = kw_block.get("keywords") if isinstance(kw_block, dict) else None
    keywords = _list_of_names(kw_list or [])

    cast_pops = [c.get("popularity") for c in cast if isinstance(c.get("popularity"), (int, float))]
    avg_cast_popularity = float(np.mean(cast_pops)) if cast_pops else None

    composers, num_composers = _extract_composers_from_crew(crew)

    return {
        "id": data.get("id"),
        "title": _none_if_empty(data.get("title")) or _none_if_empty(data.get("original_title")),
        "original_title": _none_if_empty(data.get("original_title")),
        "release_date": _none_if_empty(data.get("release_date")),
        "revenue": revenue,
        "budget": budget,
        "runtime": runtime,
        "certification": cert,
        "genres": genres,
        "production_countries": production_countries,
        "spoken_languages": spoken_languages,
        "keywords": keywords,
        "directors": directors,
        "lead_cast": lead_cast,
        "lead_cast_genders": lead_cast_genders,
        "original_language": original_language,
        "status": status,
        "collection_name": collection_name,
        "avg_cast_popularity": avg_cast_popularity,
        "homepage": homepage,
        "popularity": popularity,
        "vote_count": vote_count,
        "vote_average": vote_average,
        "num_spoken_languages": num_spoken_languages,
        "composers": composers,
        "num_composers": num_composers,
    }

def build_core_movie_frame(
    df: pd.DataFrame,
    client: TMDBClient,
    *,
    top_cast_k: int = DEFAULT_TOP_CAST_K,
    workers: int = DEFAULT_WORKERS,
    drop_thresholds: Tuple[int, int] = DEFAULT_DROP_THRESHOLDS,
    priority_countries: Sequence[str] = DEFAULT_PRIORITY_COUNTRIES,
    show_progress: bool = True,
    overwrite_title: bool = False,
) -> pd.DataFrame:
    if "id" not in df.columns:
        raise ValueError("Input DataFrame must contain 'id' column.")

    work_df = df[["id", "title"]].copy()
    work_df = work_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    ids = work_df["id"].astype(int).tolist()

    results: Dict[int, Optional[Dict[str, Any]]] = {}
    if workers <= 1:
        for mid in ids:
            results[mid] = _fetch_movie_core(
                mid,
                client,
                top_cast_k=top_cast_k,
                priority_countries=priority_countries,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _fetch_movie_core,
                    mid,
                    client,
                    top_cast_k=top_cast_k,
                    priority_countries=priority_countries,
                ): mid
                for mid in ids
            }
            for fut in as_completed(futures):
                mid = futures[fut]
                try:
                    results[mid] = fut.result()
                except Exception:
                    results[mid] = None

    rows = []
    for mid in ids:
        rec = results.get(mid)
        if rec is None:
            continue
        rows.append(rec)

    out = pd.DataFrame(rows)

    out["has_homepage"] = out["homepage"].notna().astype("int8")
    out["num_spoken_languages"] = out["num_spoken_languages"].fillna(0).astype("int16")

    budget_min, revenue_min = drop_thresholds
    if budget_min is not None:
        out = out[out["budget"].fillna(0) >= budget_min]
    if revenue_min is not None:
        out = out[out["revenue"].fillna(0) >= revenue_min]

    if not overwrite_title and "title" in df.columns:
        out = out.drop(columns=["title"])
        out = df.merge(out, on="id", how="left")

    return out

def run_core_enrichment_pipeline(
    df_union: pd.DataFrame,
    *,
    top_cast_k: int = DEFAULT_TOP_CAST_K,
    workers: int = DEFAULT_WORKERS,
    drop_thresholds: Tuple[int, int] = DEFAULT_DROP_THRESHOLDS,
    priority_countries: Sequence[str] = DEFAULT_PRIORITY_COUNTRIES,
    show_progress: bool = True,
    overwrite_title: bool = False,
) -> pd.DataFrame:
    client = TMDBClient.from_env()
    return build_core_movie_frame(
        df=df_union,
        client=client,
        top_cast_k=top_cast_k,
        workers=workers,
        drop_thresholds=drop_thresholds,
        priority_countries=priority_countries,
        show_progress=show_progress,
        overwrite_title=overwrite_title,
    )

# Feature engineering helpers and main feature function
def _slug(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s

def _log1p_safe(x) -> float:
    try:
        v = float(x)
        if not np.isfinite(v) or v < 0:
            return np.nan
        return float(np.log1p(v))
    except Exception:
        return np.nan

def _split_multi(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple, set)):
        return list(val)
    if isinstance(val, str):
        if "|" in val:
            return [v.strip() for v in val.split("|") if v.strip()]
        if val.strip() == "":
            return []
        return [val.strip()]
    return [val]

def _compute_topk_tokens_multi(ser: pd.Series, top_k: int) -> List[str]:
    freq: Dict[str, int] = {}
    for items in ser.dropna():
        for t in _split_multi(items):
            freq[t] = freq.get(t, 0) + 1
    if not freq:
        return []
    return sorted(freq, key=freq.get, reverse=True)[:top_k]

def _compute_topk_tokens_single(ser: pd.Series, top_k: int) -> List[str]:
    freq: Dict[str, int] = {}
    for v in ser.dropna():
        freq[v] = freq.get(v, 0) + 1
    if not freq:
        return []
    return sorted(freq, key=freq.get, reverse=True)[:top_k]

def _multihot_from_vocab(ser: pd.Series, prefix: str, vocab: Sequence[str]) -> pd.DataFrame:
    cols = [f"{prefix}_{_slug(t)}" for t in vocab]
    out = pd.DataFrame(0, index=ser.index, columns=cols, dtype="int8")
    token2col = {t: c for t, c in zip(vocab, cols)}
    for idx, items in ser.items():
        for t in _split_multi(items):
            if t in token2col:
                out.at[idx, token2col[t]] = 1
    return out

def _onehot_from_vocab(ser: pd.Series, prefix: str, vocab: Sequence[str]) -> pd.DataFrame:
    cols = [f"{prefix}_{_slug(t)}" for t in vocab]
    out = pd.DataFrame(0, index=ser.index, columns=cols, dtype="int8")
    token2col = {t: c for t, c in zip(vocab, cols)}

    for idx, v in ser.items():
        if v in token2col:
            out.at[idx, token2col[v]] = 1

    return out


def make_features_timeaware_splits(
    df: pd.DataFrame,
    *,
    train_end_year: int,
    val_year: int,
    test_start_year: int,
    top_n_multi: int = 20,
    top_n_orig_lang: int = 10,
    top_n_cert: int = 20,
):
    df = df.copy()

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year.astype("Int16")

    df["revenue_log1p"] = df["revenue"].apply(_log1p_safe)
    df["budget_log1p"] = df["budget"].apply(_log1p_safe)

    df["log_budget_to_revenue_ratio"] = df["budget_log1p"] - df["revenue_log1p"]

    df["has_collection"] = df["collection_name"].notna().astype("int8")
    df["has_homepage"] = df["homepage"].notna().astype("int8")
    df["has_keywords"] = df["keywords"].notna().astype("int8")

    df["num_genres"] = df["genres"].apply(lambda x: len(_split_multi(x))).astype("Int16")
    df["num_production_countries"] = df["production_countries"].apply(
        lambda x: len(_split_multi(x))
    ).astype("Int16")
    df["num_directors"] = df["directors"].apply(lambda x: len(_split_multi(x))).astype("Int16")
    df["num_lead_cast"] = df["lead_cast"].apply(lambda x: len(_split_multi(x))).astype("Int16")
    df["num_keywords"] = df["keywords"].apply(lambda x: len(_split_multi(x))).astype("Int16")

    df["x_year"] = df["release_year"]

    train_mask = df["x_year"] <= train_end_year
    val_mask = df["x_year"] == val_year
    test_mask = df["x_year"] >= test_start_year

    df_train = df.loc[train_mask].copy()
    df_val = df.loc[val_mask].copy()
    df_test = df.loc[test_mask].copy()

    vocab_genres = _compute_topk_tokens_multi(df_train["genres"], top_n_multi)
    vocab_keywords = _compute_topk_tokens_multi(df_train["keywords"], top_n_multi)
    vocab_prod_countries = _compute_topk_tokens_multi(df_train["production_countries"], top_n_multi)
    vocab_directors = _compute_topk_tokens_multi(df_train["directors"], top_n_multi)
    vocab_lead_cast = _compute_topk_tokens_multi(df_train["lead_cast"], top_n_multi)

    vocab_orig_lang = _compute_topk_tokens_single(df_train["original_language"], top_n_orig_lang)
    vocab_cert = _compute_topk_tokens_single(df_train["certification"], top_n_cert)

    def build_X(base_df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        numeric_cols = [
            "revenue_log1p",
            "budget_log1p",
            "log_budget_to_revenue_ratio",
            "runtime",
            "popularity",
            "vote_count",
            "vote_average",
            "avg_cast_popularity",
            "num_spoken_languages",
            "num_genres",
            "num_production_countries",
            "num_directors",
            "num_lead_cast",
            "num_keywords",
        ]
        parts.append(base_df[numeric_cols].astype("float32"))

        parts.append(base_df[["has_collection", "has_homepage", "has_keywords"]].astype("int8"))

        parts.append(_multihot_from_vocab(base_df["genres"], "genre", vocab_genres))
        parts.append(_multihot_from_vocab(base_df["keywords"], "kw", vocab_keywords))
        parts.append(_multihot_from_vocab(base_df["production_countries"], "pc", vocab_prod_countries))
        parts.append(_multihot_from_vocab(base_df["directors"], "dir", vocab_directors))
        parts.append(_multihot_from_vocab(base_df["lead_cast"], "cast", vocab_lead_cast))

        parts.append(_onehot_from_vocab(base_df["original_language"], "lang", vocab_orig_lang))
        parts.append(_onehot_from_vocab(base_df["certification"], "cert", vocab_cert))

        X = pd.concat(parts, axis=1)
        return X

    X_train = build_X(df_train)
    X_val = build_X(df_val)
    X_test = build_X(df_test)

    metadata = {
        "vocab_genres": vocab_genres,
        "vocab_keywords": vocab_keywords,
        "vocab_prod_countries": vocab_prod_countries,
        "vocab_directors": vocab_directors,
        "vocab_lead_cast": vocab_lead_cast,
        "vocab_orig_lang": vocab_orig_lang,
        "vocab_cert": vocab_cert,
        "train_end_year": train_end_year,
        "val_year": val_year,
        "test_start_year": test_start_year,
    }

    return X_train, X_val, X_test, metadata


def make_features_from_artifacts(df: pd.DataFrame, feat_artifacts: dict) -> pd.DataFrame:
    """
    Reproduce the SAME feature engineering as during training, but using
    saved top-K vocabularies (genres, keywords, etc.) from `topk_lists.json`.
    """
    df = df.copy()

    # Normalize feat_artifacts structure
    if "top_genres" not in feat_artifacts and "multi_vocab" in feat_artifacts:
        multi  = feat_artifacts.get("multi_vocab", {})
        single = feat_artifacts.get("single_vocab", {})

        feat_artifacts = {
            **feat_artifacts,
            "top_genres": multi.get("genres", []),
            "top_production_countries": multi.get("production_countries", []),
            "top_spoken_languages": multi.get("spoken_languages", []),
            "top_keywords": multi.get("keywords", []),
            "top_original_language": single.get("original_language", []),
            "top_certification": single.get("certification", []),
        }

    # ============================================================
    # 0) Drop unneeded columns (same as make_features)
    # ============================================================
    drop_cols = [
        "popularity",
        "vote_count",
        "vote_average",
        "collection_id",
        "collection_name",
        "avg_cast_popularity",
    ]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # ============================================================
    # 1) Numeric basics + target
    # ===============================================================
    df["y_log_revenue"] = df["revenue"].apply(lambda x: np.log1p(x) if pd.notna(x) and x >= 0 else np.nan)

    df["x_budget_log"] = df["budget"].apply(lambda x: np.log1p(x) if pd.notna(x) and x >= 0 else np.nan)
    df["x_runtime"] = pd.to_numeric(df.get("runtime"), errors="coerce")
    df["x_num_spoken_languages"] = pd.to_numeric(df.get("num_spoken_languages"), errors="coerce")

    for b in ["is_in_collection", "has_homepage"]:
        if b in df.columns:
            df[b] = df[b].astype(bool).astype(int)

    # ============================================================
    # 2) Time features
    # ============================================================
    dt = pd.to_datetime(df["release_date"], errors="coerce")
    df["x_year"] = dt.dt.year
    df["x_month"] = dt.dt.month
    df["x_weekday"] = dt.dt.weekday
    df["x_weekofyear"] = dt.dt.isocalendar().week.astype("Int64")
    df["x_day"] = dt.dt.day
    df["x_quarter"] = dt.dt.quarter

    # ============================================================
    # 3) Gender ratios from lead_cast_genders
    # ============================================================
    def _split_multi(val):
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str):
            return [p.strip() for p in val.split("|") if p.strip()]
        return []

    def _gender_counts(cell):
        tokens = [t.strip().lower() for t in _split_multi(cell)]
        return (
            tokens.count("male"),
            tokens.count("female"),
            tokens.count("non binary") + tokens.count("non-binary"),
            tokens.count("unknown"),
        )

    male, female, nb, unk = [], [], [], []
    for v in df.get("lead_cast_genders", "").fillna(""):
        m, f, n, u = _gender_counts(v)
        male.append(m); female.append(f); nb.append(n); unk.append(u)

    cnt_total = np.array(male) + np.array(female) + np.array(nb) + np.array(unk)
    cnt_total = np.where(cnt_total == 0, np.nan, cnt_total)

    df["x_cast_ratio_male"] = np.array(male) / cnt_total
    df["x_cast_ratio_female"] = np.array(female) / cnt_total
    df["x_cast_ratio_nonbinary"] = np.array(nb) / cnt_total
    df[["x_cast_ratio_male", "x_cast_ratio_female", "x_cast_ratio_nonbinary"]] = \
        df[["x_cast_ratio_male", "x_cast_ratio_female", "x_cast_ratio_nonbinary"]].fillna(0)

    # ============================================================
    # 4) One-hot using saved vocabularies (critically important!)
    # ============================================================

    def _multihot_from_vocab(series, vocab, prefix):
        out = pd.DataFrame(0, index=series.index,
                           columns=[f"{prefix}_{v}" for v in vocab], dtype=int)
        for idx, items in series.items():
            for t in _split_multi(items):
                slug = v_map.get(t)
                if slug:
                    col = f"{prefix}_{slug}"
                    if col in out.columns:
                        out.at[idx, col] = 1
        return out

    def _onehot_from_vocab(series, vocab, prefix):
        series = series.fillna("NA").astype(str)
        out = pd.DataFrame(0, index=series.index,
                           columns=[f"{prefix}_{v}" for v in vocab], dtype=int)
        for idx, val in series.items():
            slug = val.lower().strip()
            slug = slug.replace(" ", "_")
            if slug in vocab:
                col = f"{prefix}_{slug}"
                out.at[idx, col] = 1
        return out

    # Helper: slug mapping for multi-label (ensures same naming)
    def _slug(s):
        s = (s or "").lower().strip()
        s = re.sub(r"[^\w\s-]", "", s)
        s = re.sub(r"\s+", "_", s)
        return re.sub(r"_+", "_", s)

    # --- Build vocab → slug map ---
    # This allows both original token and slug to match correctly
    v_map = {}

    # Multi-label vocabs
    for key in ["genres", "production_countries", "spoken_languages", "keywords",
                "directors", "lead_cast", "composers"]:
        vocab_raw = feat_artifacts.get(f"top_{key}", [])
        vocab_slugs = [_slug(x) for x in vocab_raw]
        for raw, s in zip(vocab_raw, vocab_slugs):
            v_map[raw] = s

    # Build one-hot frames
    oh_genres = _onehot_from_vocab(df["genres"],               [_slug(v) for v in feat_artifacts["top_genres"]],              "x_genre")
    oh_countries = _onehot_from_vocab(df["production_countries"], [_slug(v) for v in feat_artifacts["top_production_countries"]], "x_country")
    oh_langs = _onehot_from_vocab(df["spoken_languages"],      [_slug(v) for v in feat_artifacts["top_spoken_languages"]],     "x_lang")
    oh_keywords = _onehot_from_vocab(df["keywords"],           [_slug(v) for v in feat_artifacts["top_keywords"]],             "x_kw")

    oh_origlang = _onehot_from_vocab(df["original_language"],  [_slug(v) for v in feat_artifacts["top_original_language"]],    "x_origlang")
    oh_cert     = _onehot_from_vocab(df["certification"].fillna("NA"),
                                     [_slug(v) for v in feat_artifacts["top_certification"]],
                                     "x_cert")

    # ============================================================
    # 5) Assemble final dataframe
    # ============================================================
    X_oh = pd.concat(
        [oh_genres, oh_countries, oh_langs, oh_keywords, oh_origlang, oh_cert],
        axis=1
    )

    df_out = pd.concat([df, X_oh], axis=1)

    # Build final order: initial cols → x_* → y_log_revenue
    initial_cols_all = [
        "id", "title", "original_title", "release_date", "revenue", "budget",
        "runtime", "certification", "genres", "production_countries",
        "spoken_languages", "keywords", "directors", "lead_cast", "composers",
        "original_language", "lead_cast_genders", "num_spoken_languages",
        "is_in_collection", "has_homepage",
    ]
    initial_cols = [c for c in initial_cols_all if c in df_out.columns]

    x_cols = [c for c in df_out.columns if c.startswith("x_")]

    ordered_cols = initial_cols + [c for c in x_cols if c != "y_log_revenue"] + ["y_log_revenue"]
    df_out = df_out[ordered_cols]

    # rename binary columns as in original make_features
    df_out = df_out.rename(columns={
        "is_in_collection": "x_is_in_collection",
        "has_homepage": "x_has_homepage"
    })

    return df_out
