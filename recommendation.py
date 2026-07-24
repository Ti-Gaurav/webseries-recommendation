"""
recommendation.py
------------------
Content-based recommendation engine for the Web Series Recommendation
System.

Approach:
1. Combine each show's genre, cast, director, language and description
   into a single "tags" string (done in preprocessing.py).
2. Vectorize all shows' tags using scikit-learn's CountVectorizer to
   build a bag-of-words matrix.
3. Compute pairwise cosine similarity between every pair of shows.
4. For a given title, return the top-N most similar shows (excluding
   itself), sorted by similarity score.

The engine is wrapped in a class so the (expensive) vectorization and
similarity matrix are computed once and reused across repeated
recommendation calls -- important for a responsive Streamlit app.
"""

from difflib import get_close_matches
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import load_and_prepare


class RecommendationEngine:
    """Content-based recommender using CountVectorizer + cosine similarity."""

    def __init__(self, data: Optional[pd.DataFrame] = None, max_features: int = 5000):
        """
        Args:
            data: Pre-cleaned dataframe with a 'tags' column. If None,
                  the dataset is loaded and prepared automatically via
                  preprocessing.load_and_prepare().
            max_features: Vocabulary cap passed to CountVectorizer to
                  keep the similarity matrix computation efficient.
        """
        self.df = data if data is not None else load_and_prepare()

        if "tags" not in self.df.columns:
            raise ValueError("Input dataframe must contain a 'tags' column.")

        # Reset index so positional (iloc) and label-based lookups agree,
        # which matters when mapping similarity-matrix rows back to titles.
        self.df = self.df.reset_index(drop=True)

        # A lowercase title -> row index map for fast, case-insensitive lookup.
        self._title_index = {
            title.lower(): idx for idx, title in enumerate(self.df["title"])
        }

        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self._vectors = self.vectorizer.fit_transform(self.df["tags"]).toarray()

        # Precompute the full cosine similarity matrix once. For a few
        # hundred titles this is a tiny, fast matrix (N x N floats).
        self.similarity_matrix = cosine_similarity(self._vectors)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_titles(self) -> List[str]:
        """Return a sorted list of all available (unique) show titles."""
        return sorted(self.df["title"].tolist())

    def title_exists(self, title: str) -> bool:
        return title.strip().lower() in self._title_index

    def find_closest_titles(self, query: str, n: int = 5) -> List[str]:
        """
        Fuzzy-match a possibly misspelled/partial title against the
        known catalog, e.g. "Money Heest" -> ["Money Heist"].
        Used to gracefully handle incorrect or missing titles.
        """
        query = query.strip().lower()
        if not query:
            return []

        all_titles_lower = list(self._title_index.keys())

        # First try substring matches (handles partial titles typed by the user)
        substring_matches = [t for t in all_titles_lower if query in t]
        if substring_matches:
            ranked = sorted(substring_matches, key=len)[:n]
            return [self.df.loc[self._title_index[t], "title"] for t in ranked]

        # Fall back to fuzzy matching for typos
        close = get_close_matches(query, all_titles_lower, n=n, cutoff=0.5)
        return [self.df.loc[self._title_index[t], "title"] for t in close]

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return the top_n most similar web series to `title` as a
        DataFrame (excluding the title itself), sorted by similarity.

        Raises:
            ValueError: if the title is empty, or not found in the
                        catalog (even after fuzzy matching).
        """
        if not title or not title.strip():
            raise ValueError("Please provide a non-empty title.")

        key = title.strip().lower()

        if key not in self._title_index:
            suggestions = self.find_closest_titles(title)
            if suggestions:
                raise ValueError(
                    f"'{title}' was not found. Did you mean: {', '.join(suggestions)}?"
                )
            raise ValueError(
                f"'{title}' was not found in the catalog and no close matches exist."
            )

        idx = self._title_index[key]
        scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score, descending; skip index 0 (the show itself)
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx]
        top_scores = scores[:top_n]

        result_indices = [i for i, _ in top_scores]
        result_similarity = [round(float(sim), 4) for _, sim in top_scores]

        result_df = self.df.loc[result_indices].copy()
        result_df["similarity"] = result_similarity
        result_df = result_df.reset_index(drop=True)

        return result_df[[
            "title", "genre", "cast", "director", "language",
            "platform", "release_year", "imdb_rating", "description", "similarity",
        ]]


# ----------------------------------------------------------------------
# Module-level singleton loader (used by app.py via st.cache_resource)
# ----------------------------------------------------------------------

def build_engine() -> RecommendationEngine:
    """Factory function used by the Streamlit app to build the engine."""
    return RecommendationEngine()


if __name__ == "__main__":
    # Quick manual smoke test when running this file directly:
    #   python recommendation.py
    engine = build_engine()
    print(f"Engine built with {len(engine.df)} titles.\n")

    sample_title = "Breaking Bad"
    print(f"Top 10 recommendations for '{sample_title}':\n")
    recs = engine.recommend(sample_title, top_n=10)
    for _, row in recs.iterrows():
        print(f" - {row['title']} (similarity={row['similarity']}, genre={row['genre']})")

    print("\nTesting graceful handling of a misspelled title ('Money Heest'):")
    try:
        engine.recommend("Money Heest")
    except ValueError as e:
        print(f" -> Handled gracefully: {e}")
