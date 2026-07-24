"""
app.py
------
Streamlit front-end for the Web Series Recommendation System.

Features:
- Netflix-inspired dark theme
- Search bar + dropdown for selecting a show
- "Recommend" button that triggers the content-based engine
- Responsive poster cards showing rating, language, platform,
  release year, and description
- Sidebar with app info / dataset stats / how-it-works
- Graceful error handling for missing/misspelled titles
- Loading spinner while recommendations are computed
"""

import streamlit as st

from recommendation import build_engine
from tmdb import fetch_poster_url, is_configured as tmdb_is_configured

# ----------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Web Series Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Netflix-inspired dark theme (custom CSS)
# ----------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* ---- Global background & fonts ---- */
    .stApp {
        background-color: #141414;
        color: #f5f5f1;
    }

    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }

    /* ---- Header / hero banner ---- */
    .hero-banner {
        padding: 1.75rem 2rem;
        border-radius: 12px;
        background: linear-gradient(120deg, #1f1f1f 0%, #000000 60%, #300000 100%);
        border: 1px solid #2a2a2a;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #E50914;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #b3b3b3;
        font-size: 1.05rem;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f5f5f1;
        border-left: 5px solid #E50914;
        padding-left: 0.6rem;
        margin: 1.4rem 0 1rem 0;
    }

    /* ---- Recommendation cards ---- */
    .series-card {
        background-color: #1c1c1c;
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 1rem;
        border: 1px solid #2a2a2a;
        transition: transform 0.15s ease, border-color 0.15s ease;
        height: 100%;
    }
    .series-card:hover {
        transform: translateY(-4px);
        border-color: #E50914;
    }
    .series-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0 0.3rem 0;
        min-height: 2.6em;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.5rem;
    }
    .badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        background-color: #2a2a2a;
        color: #d2d2d2;
        white-space: nowrap;
    }
    .badge-rating {
        background-color: #E50914;
        color: white;
    }
    .series-desc {
        font-size: 0.82rem;
        color: #a3a3a3;
        line-height: 1.35;
        max-height: 6.2em;
        overflow: hidden;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #0b0b0b;
        border-right: 1px solid #2a2a2a;
    }

    /* ---- Buttons ---- */
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.55rem 1.4rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #b90710;
        color: white;
    }

    /* ---- Inputs ---- */
    div[data-baseweb="select"] > div, .stTextInput>div>div>input {
        background-color: #1f1f1f;
        color: #f5f5f1;
        border-color: #333333;
    }

    footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached resource loading
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_engine():
    """Build (and cache) the recommendation engine once per session."""
    return build_engine()


# ----------------------------------------------------------------------
# Helper rendering functions
# ----------------------------------------------------------------------

def render_hero():
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🎬 Web Series Recommendation System</div>
            <div class="hero-subtitle">
                Discover your next binge-worthy show — powered by content-based
                filtering using genre, cast, director, language and plot similarity.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(engine, total_titles: int):
    with st.sidebar:
        st.markdown("## 🎥 About")
        st.write(
            "This app recommends web series similar to one you already "
            "love, using a **content-based filtering** model built with "
            "`CountVectorizer` and **cosine similarity** over each show's "
            "genre, cast, director, language and description."
        )

        st.markdown("---")
        st.markdown("## 📊 Dataset")
        st.metric("Total unique series", total_titles)
        genres = engine.df["genre"].str.split(",").explode().str.strip()
        st.metric("Unique genres", genres.nunique())
        platforms = engine.df["platform"].str.split(",").explode().str.strip()
        st.metric("Streaming platforms", platforms.nunique())

        st.markdown("---")
        st.markdown("## ⚙️ How it works")
        st.markdown(
            """
            1. Pick or search for a series you like
            2. Click **Recommend**
            3. We vectorize genre + cast + director + language +
               description for every show
            4. Cosine similarity ranks the most similar shows
            5. Top 10 matches are displayed as cards with posters
            """
        )

        st.markdown("---")
        if tmdb_is_configured():
            st.success("TMDB API key detected — live posters enabled.")
        else:
            st.warning(
                "No TMDB API key found — placeholder posters will be shown. "
                "See README.md to configure one."
            )

        st.markdown("---")
        st.caption("Built with Streamlit, Pandas, Scikit-learn & TMDB API.")


def render_series_card(row):
    """Render a single recommended series as an HTML/Streamlit hybrid card."""
    poster_url = fetch_poster_url(row["title"], int(row["release_year"]))

    with st.container():
        st.markdown('<div class="series-card">', unsafe_allow_html=True)
        st.image(poster_url, use_container_width=True)
        st.markdown(f'<div class="series-title">{row["title"]}</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="badge-row">
                <span class="badge badge-rating">⭐ {row['imdb_rating']}</span>
                <span class="badge">📅 {row['release_year']}</span>
                <span class="badge">🌐 {row['language']}</span>
                <span class="badge">📺 {row['platform']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        description = row["description"]
        if len(description) > 160:
            description = description[:157].rstrip() + "..."
        st.markdown(f'<div class="series-desc">{description}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def render_recommendations(recs_df):
    st.markdown('<div class="section-header">Because you liked this, you might enjoy:</div>', unsafe_allow_html=True)

    cols_per_row = 5
    rows = [recs_df.iloc[i:i + cols_per_row] for i in range(0, len(recs_df), cols_per_row)]

    for chunk in rows:
        cols = st.columns(cols_per_row)
        for col, (_, series_row) in zip(cols, chunk.iterrows()):
            with col:
                render_series_card(series_row)


# ----------------------------------------------------------------------
# Main app logic
# ----------------------------------------------------------------------

def main():
    render_hero()

    # ---- Load engine with error handling ----
    try:
        with st.spinner("Loading dataset and building recommendation model..."):
            engine = get_engine()
    except FileNotFoundError as e:
        st.error(f"❌ Dataset error: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"❌ Dataset validation error: {e}")
        st.stop()
    except Exception as e:  # noqa: BLE001 - top-level safety net for the UI
        st.error(f"❌ Unexpected error while loading the app: {e}")
        st.stop()

    all_titles = engine.get_all_titles()
    render_sidebar(engine, len(all_titles))

    # ---- Search + selection controls ----
    st.markdown('<div class="section-header">Find a series you love</div>', unsafe_allow_html=True)

    search_col, dropdown_col = st.columns([1, 1])

    with search_col:
        search_query = st.text_input(
            "🔍 Search by title",
            placeholder="e.g. Breaking Bad, Money Heist, Squid Game...",
            help="Type a title (even partial or misspelled) to filter the dropdown below.",
        )

    # Filter dropdown options based on the search box, if used
    if search_query.strip():
        filtered = engine.find_closest_titles(search_query, n=25) or [
            t for t in all_titles if search_query.lower() in t.lower()
        ]
        dropdown_options = filtered if filtered else all_titles
    else:
        dropdown_options = all_titles

    with dropdown_col:
        selected_title = st.selectbox(
            "🎯 Or pick a series from the dropdown",
            options=dropdown_options,
            index=0 if dropdown_options else None,
        )

    recommend_clicked = st.button("🚀 Recommend Similar Series", use_container_width=True)

    st.markdown("---")

    # ---- Recommendation flow ----
    if recommend_clicked:
        query_title = selected_title or search_query

        if not query_title or not query_title.strip():
            st.error("❌ Please search for or select a series title first.")
            return

        try:
            with st.spinner(f"Finding series similar to '{query_title}'..."):
                recommendations = engine.recommend(query_title, top_n=10)
        except ValueError as e:
            # Handles: empty title, title not found, and shows fuzzy-match
            # suggestions gracefully instead of crashing.
            st.error(f"❌ {e}")
            return
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ Something went wrong while generating recommendations: {e}")
            return

        if recommendations.empty:
            st.warning("No similar series were found for that title.")
            return

        render_recommendations(recommendations)
    else:
        st.info("👈 Search or select a web series above, then click **Recommend Similar Series** to get started.")


if __name__ == "__main__":
    main()
