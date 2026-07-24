"""
preprocessing.py
-----------------
Handles loading, cleaning, and feature engineering for the web series
dataset. The output of this module is a "tags" column that combines
genre, cast, director, language and description into a single bag of
words used by the recommendation engine (recommendation.py).

Keeping this logic separate from app.py and recommendation.py keeps
each file focused on a single responsibility (data prep vs. modeling
vs. UI), which makes the project easier to test and maintain.
"""

import os
import re

import pandas as pd

# Path to the dataset, resolved relative to this file so the app works
# regardless of the working directory it is launched from.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "webseries.csv")

REQUIRED_COLUMNS = [
    "title",
    "genre",
    "cast",
    "director",
    "language",
    "platform",
    "release_year",
    "imdb_rating",
    "description",
]


def _clean_text(value: str) -> str:
    """Lowercase, strip punctuation/extra whitespace from a text field."""
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _tokens_no_space(value: str) -> str:
    """
    Turn a comma-separated field (e.g. "Bryan Cranston, Aaron Paul")
    into tokens with internal spaces removed (e.g. "bryancranston
    aaronpaul"), so that "Bryan" from one actor's first name doesn't
    get conflated with "Bryan" from an unrelated actor in the
    vectorizer's vocabulary.
    """
    if pd.isna(value):
        return ""
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    return " ".join(p.lower().replace(" ", "") for p in parts)


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame, raising a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Make sure data/webseries.csv exists."
        )
    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
    - Drop rows with a missing/blank title (can't recommend or display those)
    - Drop duplicate titles (case-insensitive), keeping the first occurrence
    - Fill missing values in optional descriptive columns with safe defaults
    - Ensure numeric columns are actually numeric
    """
    df = df.copy()

    # Drop rows with no title at all
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]

    # Normalize title whitespace
    df["title"] = df["title"].astype(str).str.strip()

    # Drop duplicate entries (case-insensitive) - keep first occurrence
    df["_title_lower"] = df["title"].str.lower()
    df = df.drop_duplicates(subset="_title_lower", keep="first")
    df = df.drop(columns="_title_lower")

    # Fill missing text fields with neutral placeholders
    text_cols = ["genre", "cast", "director", "language", "platform", "description"]
    for col in text_cols:
        df[col] = df[col].fillna("Unknown")
        df[col] = df[col].replace("", "Unknown")

    # Coerce numeric columns, fill missing with dataset median as a safe default
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["release_year"] = df["release_year"].fillna(df["release_year"].median()).astype(int)

    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
    df["imdb_rating"] = df["imdb_rating"].fillna(df["imdb_rating"].median()).round(1)

    df = df.reset_index(drop=True)
    return df


def build_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single "tags" column per row that concatenates the features
    used for content-based similarity: genre, cast, director, language
    and description. This bag-of-words column is what CountVectorizer
    will be fit on inside recommendation.py.
    """
    df = df.copy()

    df["tags"] = (
        df["genre"].apply(_clean_text) + " "
        + df["cast"].apply(_tokens_no_space) + " "
        + df["director"].apply(_tokens_no_space) + " "
        + df["language"].apply(_clean_text) + " "
        + df["description"].apply(_clean_text)
    )

    # Genre and cast matter more for "similar show" recommendations than a
    # loosely-worded synopsis, so we duplicate them to weight them higher
    # inside the CountVectorizer's bag-of-words model.
    df["tags"] = (
        df["genre"].apply(_clean_text) + " "
        + df["genre"].apply(_clean_text) + " "
        + df["cast"].apply(_tokens_no_space) + " "
        + df["cast"].apply(_tokens_no_space) + " "
        + df["director"].apply(_tokens_no_space) + " "
        + df["language"].apply(_clean_text) + " "
        + df["description"].apply(_clean_text)
    )

    return df


def load_and_prepare(path: str = DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load -> clean -> build tags in one call."""
    df = load_raw_data(path)
    df = clean_data(df)
    df = build_tags(df)
    return df


if __name__ == "__main__":
    # Quick manual sanity check when running this file directly.
    data = load_and_prepare()
    print(f"Loaded {len(data)} unique series.")
    print(data[["title", "tags"]].head(3))
