import os
import time
from typing import Dict, Optional, Tuple, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv


class TMDBClient:
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str, language: str = "en-US", sleep_between_calls: float = 0.3):
        """
        Parameters
        ----------
        api_key : str
            TMDB API v3 key.
        language : str
            Default language for queries (advisory; some endpoints ignore).
        sleep_between_calls : float
            Polite delay to respect TMDB's rate limits.
        """
        if not api_key:
            raise ValueError("TMDB API key is missing.")
        self.api_key = api_key
        self.language = language
        self.sleep = sleep_between_calls

        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    @classmethod
    def from_env(cls, env_key: str = "TMDB_API_KEY", **kwargs) -> "TMDBClient":
        """
        Create a TMDBClient using an API key from environment / .env.

        Priority order:
          1) env_key argument (default: TMDB_API_KEY)
          2) TMDB_KEY
          3) TMDB_API_KEY  (if env_key != TMDB_API_KEY)
        """
        # Load .env if present (no-op if already loaded)
        load_dotenv()

        # Build candidate env var names
        candidates = [env_key]
        if "TMDB_KEY" not in candidates:
            candidates.append("TMDB_KEY")
        if "TMDB_API_KEY" not in candidates:
            candidates.append("TMDB_API_KEY")

        api_key = None
        for name in candidates:
            value = os.getenv(name)
            if value:
                api_key = value
                break

        if not api_key:
            raise RuntimeError(
                "TMDB API key not found. Please set one of: "
                f"{', '.join(candidates)} in your environment or .env file."
            )

        return cls(api_key=api_key, **kwargs)

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        params = dict(params or {})
        params["api_key"] = self.api_key
        if "language" not in params and self.language:
            params["language"] = self.language

        url = f"{self.BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=30)

        # simple extra handling for 429 beyond Retry
        if resp.status_code == 429:
            time.sleep(1.2)
            resp = self.session.get(url, params=params, timeout=30)

        resp.raise_for_status()
        time.sleep(self.sleep)
        return resp.json()

    # ---- High-level endpoints ----

    def discover_movies(self, year: int, page: int, include_adult: bool = False, sort_by: str = "popularity.desc") -> Dict:
        """
        sort_by examples: "popularity.desc" (default), "revenue.desc", "vote_count.desc"
        """
        return self._get(
            "/discover/movie",
            {
                "include_adult": str(include_adult).lower(),
                "primary_release_year": year,
                "sort_by": sort_by,
                "page": page,
            },
        )

    def movie_details(self, movie_id: int) -> Dict:
        # details + append extras (release_dates, keywords)
        return self._get(
            f"/movie/{movie_id}",
            {"append_to_response": "release_dates,keywords"},
        )

    def movie_details_min(self, movie_id: int) -> Dict:
        # light details only (fast): id, title, budget, revenue, release_date, status, etc.
        return self._get(f"/movie/{movie_id}", {})

    def movie_credits(self, movie_id: int) -> Dict:
        return self._get(f"/movie/{movie_id}/credits")

    # ---- Parsing helpers ----

    @staticmethod
    def parse_certification(release_dates: Dict, priority_countries: Tuple[str, ...] = ("US", "GB")) -> Optional[str]:
        if not release_dates or "results" not in release_dates:
            return None
        country_map = {r.get("iso_3166_1"): r.get("release_dates", []) for r in release_dates.get("results", [])}

        def pick_cert(entries):
            if not entries:
                return None
            theatrical = [e for e in entries if e.get("type") == 3 and e.get("certification")]
            if theatrical:
                return theatrical[0].get("certification") or None
            for e in entries:
                if e.get("certification"):
                    return e.get("certification")
            return None

        for c in priority_countries:
            cert = pick_cert(country_map.get(c, []))
            if cert:
                return cert
        for entries in country_map.values():
            cert = pick_cert(entries)
            if cert:
                return cert
        return None

    @staticmethod
    def top_people(credits: Dict, role_key: str, role_value: str, top_k: int = 3) -> List[str]:
        if not credits:
            return []
        if role_key == "crew":
            crew = credits.get("crew", []) or []
            names = [c.get("name") for c in crew if (c.get("job") or "").lower() == role_value.lower()]
            seen, out = set(), []
            for n in names:
                if n and n not in seen:
                    seen.add(n)
                    out.append(n)
            return out[:top_k]
        elif role_key == "cast":
            cast = credits.get("cast", []) or []
            names = [c.get("name") for c in sorted(cast, key=lambda x: (x.get("order") is None, x.get("order")))]
            seen, out = set(), []
            for n in names:
                if n and n not in seen:
                    seen.add(n)
                    out.append(n)
            return out[:top_k]
        return []

    @staticmethod
    def extract_keywords(details: Dict, top_k: Optional[int] = None) -> List[str]:
        if not details:
            return []
        kw = details.get("keywords", {})
        items = kw.get("keywords") or kw.get("results") or []
        names = [k.get("name") for k in items if k.get("name")]
        if top_k and len(names) > top_k:
            names = names[:top_k]
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    # ---- Aggregation ----

    def aggregate_movie_record(
        self,
        movie_id: int,
        certification_priority: Tuple[str, ...] = ("US", "GB"),
        top_cast_k: int = 3,
        top_kw_k: int = 300,
    ) -> Dict:
        details = self.movie_details(movie_id)
        credits = self.movie_credits(movie_id)

        genres = [g["name"] for g in details.get("genres", []) if g.get("name")]
        prod_countries = [c["iso_3166_1"] for c in details.get("production_countries", []) if c.get("iso_3166_1")]
        languages = [l.get("english_name") or l.get("name") for l in details.get("spoken_languages", [])]

        certification = self.parse_certification(details.get("release_dates"), certification_priority)
        directors = self.top_people(credits, role_key="crew", role_value="Director", top_k=5)
        lead_cast = self.top_people(credits, role_key="cast", role_value="", top_k=top_cast_k)
        keywords = self.extract_keywords(details, top_k=top_kw_k)

        return {
            "id": details.get("id"),
            "title": details.get("title"),
            "original_title": details.get("original_title"),
            "release_date": details.get("release_date"),
            "revenue": details.get("revenue"),
            "budget": details.get("budget"),
            "runtime": details.get("runtime"),
            "popularity": details.get("popularity"),
            "vote_count": details.get("vote_count"),
            "vote_average": details.get("vote_average"),
            "collection_id": (details.get("belongs_to_collection") or {}).get("id"),
            "certification": certification,
            "genres": genres,
            "production_countries": prod_countries,
            "spoken_languages": languages,
            "keywords": keywords,
            "directors": directors,
            "lead_cast": lead_cast,
            "imdb_id": details.get("imdb_id"),
            "homepage": details.get("homepage"),
            "status": details.get("status"),
            "original_language": details.get("original_language"),
        }
