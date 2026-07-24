# 🎬 Web Series Recommendation System

A production-quality, Netflix-inspired **Web Series Recommendation System** built with
Python and Streamlit. It uses **content-based filtering** (`CountVectorizer` +
**cosine similarity**) over genre, cast, director, language and plot description
to recommend the top 10 shows most similar to one you already love — complete
with live posters pulled from **TMDB**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 🎨 Modern **Netflix-inspired dark theme** UI
- 🔍 Search bar with fuzzy title matching (handles typos/partial titles)
- 🎯 Dropdown selector synced with the search bar
- 🚀 One-click **Recommend** button
- 🖼️ Live poster fetching via the **TMDB API** (with graceful placeholder fallback)
- ⭐ IMDb rating, 🌐 language, 📺 platform, 📅 release year, and description on every card
- 📱 Responsive card grid (5 → fewer columns automatically on narrower screens)
- 📚 Informative sidebar with dataset stats and "how it works"
- 🛡️ Robust error handling for missing / misspelled / empty titles
- ⏳ Loading spinners while the model builds and while recommendations compute
- 🧹 Automatic de-duplication of dataset entries

---

## 🧠 How the Recommendation Engine Works

1. **`preprocessing.py`** loads `data/webseries.csv`, cleans it (drops rows with
   no title, removes duplicate titles, fills missing values), and builds a single
   `tags` column per show that combines **genre (×2 weight) + cast (×2 weight) +
   director + language + description**.
2. **`recommendation.py`** fits a `sklearn.feature_extraction.text.CountVectorizer`
   over every show's `tags` to build a bag-of-words matrix, then computes an
   `N x N` **cosine similarity** matrix across all shows.
3. When a user picks a title, the engine looks up that show's similarity row,
   sorts all other shows by similarity score, and returns the **top 10 matches**.
4. If a title isn't found, `difflib.get_close_matches` (plus substring matching)
   suggests the closest real titles instead of crashing — e.g. searching
   `"Money Heest"` suggests `"Money Heist"`.
5. **`tmdb.py`** fetches a poster image for each recommended title from TMDB,
   falling back to a placeholder image if no API key is configured or the
   request fails for any reason.

---

## 📁 Folder Structure

```
webseries-recommendation/
│
├── app.py                       # Streamlit UI (entry point)
├── recommendation.py            # Content-based recommendation engine
├── tmdb.py                      # TMDB API wrapper (poster fetching)
├── preprocessing.py             # Data loading, cleaning, feature engineering
├── generate_dataset.py          # One-time script that generated webseries.csv
│
├── data/
│   └── webseries.csv            # 251 unique web series (title, genre, cast, ...)
│
├── assets/                      # Static assets (logos, icons, etc.)
│   └── .gitkeep
│
├── screenshots/                 # App screenshots for documentation
│   └── .gitkeep
│
├── .streamlit/
│   ├── config.toml              # Dark theme configuration
│   └── secrets.toml.example     # Template for your local TMDB API key
│
├── requirements.txt             # Python dependencies
├── README.md                    # You are here
└── .gitignore                   # Files/folders excluded from git
```

---

## 🛠️ Tech Stack

| Layer               | Technology                          |
|---------------------|--------------------------------------|
| Language             | Python 3.12                        |
| UI Framework         | Streamlit                          |
| Data handling         | Pandas, NumPy                     |
| ML / Similarity       | Scikit-learn (`CountVectorizer`, `cosine_similarity`) |
| HTTP client            | Requests                         |
| Poster images           | TMDB API                       |
| Version control          | Git                           |

---

## 📦 Installation Instructions

### 1. Prerequisites
- Python **3.12** (or 3.10+) installed
- `pip` and `git` available on your PATH

### 2. Clone the repository
```bash
git clone https://github.com/<your-username>/webseries-recommendation.git
cd webseries-recommendation
```

### 3. Create and activate a virtual environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 How to Obtain and Configure a TMDB API Key

Poster images are optional — the app runs fine without a key (it shows a
placeholder poster) — but here's how to enable real posters:

1. Create a free account at [themoviedb.org](https://www.themoviedb.org/signup).
2. Go to **Settings → API** ([direct link](https://www.themoviedb.org/settings/api))
   and request an **API key** (choose "Developer" — it's free and instant for
   personal projects).
3. Configure the key using **either** of these two methods:

   **Option A — Environment variable (recommended for local dev):**
   ```bash
   # macOS / Linux
   export TMDB_API_KEY="your_tmdb_api_key_here"

   # Windows (PowerShell)
   $env:TMDB_API_KEY="your_tmdb_api_key_here"
   ```

   **Option B — Streamlit secrets file (recommended for deployment):**
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Then edit `.streamlit/secrets.toml` and paste your key:
   ```toml
   TMDB_API_KEY = "your_tmdb_api_key_here"
   ```
   This file is already listed in `.gitignore`, so your key will never be
   committed to version control.

The sidebar of the running app will tell you whether a key was detected.

---

## ▶️ How to Run the Project Locally

```bash
streamlit run app.py
```

Then open the URL Streamlit prints in your terminal (usually
`http://localhost:8501`) in your browser.

---

## 🌱 Git Commands

### Initialize and push a new repository
```bash
git init
git add .
git commit -m "Initial commit: Web Series Recommendation System"
git branch -M main
git remote add origin https://github.com/<your-username>/webseries-recommendation.git
git push -u origin main
```

### Everyday workflow
```bash
git status
git add .
git commit -m "Describe your change here"
git push
```

### Working with branches
```bash
git checkout -b feature/collaborative-filtering
# ... make changes ...
git add .
git commit -m "Add collaborative filtering module"
git push -u origin feature/collaborative-filtering
```

---

## 🧪 Regenerating the Dataset (optional)

`data/webseries.csv` is already committed, but if you want to see how it was
built or extend it, run:
```bash
python generate_dataset.py
```
This reads the curated list inside `generate_dataset.py`, de-duplicates it,
and rewrites `data/webseries.csv`.

## 🧪 Testing the engine standalone (optional)
```bash
python recommendation.py
```
This prints the top 10 recommendations for "Breaking Bad" and demonstrates
graceful handling of a misspelled title, without needing to launch Streamlit.

---

## 🚧 Error Handling Summary

| Scenario                          | Behavior                                                |
|------------------------------------|----------------------------------------------------------|
| Empty search / no title selected   | Friendly inline error, no crash                          |
| Misspelled / unknown title         | Fuzzy-matched suggestions shown to the user               |
| Dataset file missing                | Clear `FileNotFoundError` message on app startup           |
| Duplicate rows in the CSV            | Automatically de-duplicated during preprocessing           |
| TMDB API key missing / request fails  | Falls back to a placeholder poster, app keeps working    |

---

## 🚀 Suggestions for Future Improvements

- **Hybrid recommendations**: Blend content-based scores with collaborative
  filtering (e.g. using user ratings/watch history) for personalized results.
- **TF-IDF weighting**: Swap or combine `CountVectorizer` with `TfidfVectorizer`
  to down-weight overly common terms across descriptions.
- **User accounts & watchlists**: Let users save shows, mark "already watched",
  and get recommendations that exclude those.
- **Explainable recommendations**: Show *why* a series was recommended (e.g.
  "shares 3 cast members and the Crime/Drama genre").
- **Trailer integration**: Embed TMDB/YouTube trailers directly on each card.
- **Multi-language UI**: Localize the interface for non-English audiences.
- **Larger, live-updated dataset**: Periodically sync `webseries.csv` with the
  TMDB "Popular TV" endpoint instead of a static file.
- **Deploy**: Ship to Streamlit Community Cloud, Hugging Face Spaces, or Docker
  + a cloud provider for public access.
- **Unit tests**: Add `pytest` coverage for `preprocessing.py` and
  `recommendation.py` (duplicate handling, missing-title errors, similarity
  ranking correctness).
- **Caching layer**: Cache TMDB poster responses to disk/DB to reduce API
  calls across app restarts (currently cached only in-memory per session).

---

## 📄 License

This project is provided for educational purposes. Feel free to fork, modify,
and build on top of it.

## 🙏 Acknowledgements

- [TMDB](https://www.themoviedb.org/) for the poster/metadata API
- [Streamlit](https://streamlit.io/) for the app framework
- [Scikit-learn](https://scikit-learn.org/) for `CountVectorizer` and
  `cosine_similarity`
