import re
import json
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="centered")

STOP_WORDS = set(ENGLISH_STOP_WORDS)

def extract_names(json_str):
    if pd.isna(json_str) or not isinstance(json_str, str):
        return ""
    try:
        items = json.loads(json_str)
        return " ".join([item["name"].replace(" ", "") for item in items])
    except Exception:
        return ""

def clean_text_pipeline(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

@st.cache_resource
def build_vectorizer():
    df = pd.read_csv(
        "tmdb_5000_movies.csv",
        usecols=["title", "overview", "genres", "keywords", "tagline", "vote_average"]
    )
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    df["overview"] = df["overview"].fillna("")
    df["tagline"] = df["tagline"].fillna("")
    df["genres_clean"] = df["genres"].apply(extract_names)
    df["keywords_clean"] = df["keywords"].apply(extract_names)

    combined = (
        df["overview"] + " " +
        df["genres_clean"] + " " +
        df["keywords_clean"] + " " +
        df["tagline"]
    )
    clean_text = combined.apply(clean_text_pipeline)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(clean_text)

    # Keep only the 3 columns needed for display to save RAM
    display_df = df[["title", "vote_average", "overview"]].copy()
    return display_df, tfidf_matrix

df, tfidf_matrix = build_vectorizer()

def recommend(item_name, top_n=5):
    idx = df[df["title"] == item_name].index[0]
    # Compute cosine similarity for ONLY the selected movie (uses <1 MB RAM)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][1 : top_n + 1]

    results = []
    for movie_idx in top_indices:
        results.append({
            "Movie Title": df.loc[movie_idx, "title"],
            "Rating": df.loc[movie_idx, "vote_average"],
            "Similarity": f"{sim_scores[movie_idx]:.2%}",
            "Overview": str(df.loc[movie_idx, "overview"])[:160] + "..."
        })
    return pd.DataFrame(results)

# --- Streamlit UI ---
st.title("🎬 Content-Based Movie Recommendation System")
st.write("Select a movie to generate recommendations based on **TF-IDF** and **Cosine Similarity**.")

selected_movie = st.selectbox("Select a Movie:", sorted(df["title"].tolist()))
top_n = st.slider("Number of Recommendations:", min_value=3, max_value=10, value=5)

if st.button("Get Recommendations"):
    recommendations = recommend(selected_movie, top_n=top_n)
    st.subheader(f"Top {top_n} Recommended Movies for '{selected_movie}'")
    st.dataframe(recommendations, use_container_width=True, hide_index=True)