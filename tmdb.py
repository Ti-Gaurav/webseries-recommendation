"""
tmdb.py
-------
Thin wrapper around The Movie Database (TMDB) API, used to fetch show
poster images by title. Network calls are isolated here so the rest of
the app (recommendation.py, app.py) never has to deal with HTTP details
or API failures directly.

Get a free API key at: https://www.themoviedb.org/settings/api

The API key is read from an environment variable (TMDB_API_KEY) or a
Streamlit secret (st.secrets["TMDB_API_KEY"]) -- see README.md for
setup instructions. If no key is configured, or a request fails for
any reason (network issue, rate limit, title not found), this module
falls back to a placeholder image so the app never crashes because of
a missing poster.
"""

import os
from functools import lru_cache
from typing import Optional

import requests

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/tv"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
REQUEST_TIMEOUT_SECONDS = 6

# A neutral placeholder poster shown whenever TMDB can't return a real one.
PLACEHOLDER_POSTER_URL = (
    "https://placehold.co/500x750/141414/E50914?text=No+Poster+Available"
)


def _get_api_key() -> Optional[str]:
    """
    Resolve the TMDB API key from, in order of preference:
    1. The TMDB_API_KEY environment variable
    2. Streamlit's secrets manager (st.secrets["TMDB_API_KEY"]), if available
    """
    api_key = os.environ.get("TMDB_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st  # imported lazily to keep this module usable stand-alone
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        # streamlit not running / no secrets.toml configured -- that's fine,
        # we just fall through to returning None.
        pass

    return None


@lru_cache(maxsize=256)
def fetch_poster_url(title: str, release_year: Optional[int] = None) -> str:
    """
    Fetch a poster image URL for the given show title from TMDB.

    Args:
        title: The web series title to search for.
        release_year: Optional release year to disambiguate shows that
                       share a title (improves match accuracy).

    Returns:
        A URL string pointing to the poster image, or a placeholder
        image URL if no poster could be found or an error occurred.

    This function never raises -- any failure (missing API key, network
    error, no results, timeout) results in the placeholder URL so the
    Streamlit UI can always render something.
    """
    api_key = _get_api_key()
    if not api_key:
        return PLACEHOLDER_POSTER_URL

    params = {"api_key": api_key, "query": title}
    if release_year:
        params["first_air_date_year"] = release_year

    try:
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            return PLACEHOLDER_POSTER_URL

        poster_path = results[0].get("poster_path")
        if not poster_path:
            return PLACEHOLDER_POSTER_URL

        return f"{TMDB_IMAGE_BASE_URL}{poster_path}"

    except (requests.RequestException, ValueError, KeyError, IndexError):
        # Covers connection errors, timeouts, bad JSON, unexpected shape, etc.
        return PLACEHOLDER_POSTER_URL


def is_configured() -> bool:
    """Whether a TMDB API key is currently available."""
    return _get_api_key() is not None


if __name__ == "__main__":
    # Quick manual smoke test:  python tmdb.py
    if not is_configured():
        print("No TMDB_API_KEY configured -- fetch_poster_url() will return placeholders.")
    print(fetch_poster_url("Breaking Bad", 2008))
