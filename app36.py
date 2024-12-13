
import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import zipfile
import os

# Base URL for movie posters
BASE_URL = 'https://liangfgithub.github.io/MovieImages/'

# Load datasets
@st.cache_data
def load_movies():
    """Load the movie metadata."""
    movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
    movies = pd.read_csv(
        movies_url, 
        sep='::', 
        engine='python', 
        encoding="ISO-8859-1", 
        header=None
    )
    movies.columns = ['MovieID', 'Title', 'Genres']
    return movies

@st.cache_data
def extract_and_load_ratings(zip_path="Rmat.csv.zip"):
    """Extract and load the rating matrix from a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = z.namelist()[0]
        z.extract(csv_name)
    ratings = pd.read_csv(csv_name, index_col=0)
    os.remove(csv_name)
    return ratings

@st.cache_data
def load_similarity_matrix():
    """Load the similarity matrix for collaborative filtering."""
    return pd.read_csv("similarity_matrix_top_30.csv", index_col=0)

# Collaborative Filtering Function
def myIBCF(newuser):
    """Item-Based Collaborative Filtering."""
    S = load_similarity_matrix()
    predictions = {}
    for i in newuser.index:
        if not pd.isna(newuser[i]):
            continue
        sim_values = S.loc[i]
        relevant_movies = sim_values[sim_values.notna() & newuser.notna()]
        ratings = newuser[relevant_movies.index]
        numerator = np.sum(relevant_movies * ratings)
        denominator = np.sum(relevant_movies)
        predictions[i] = numerator / denominator if denominator > 0 else np.nan
    predictions_df = pd.DataFrame.from_dict(predictions, orient="index", columns=["PredictedRating"])
    predictions_df = predictions_df.sort_values(by="PredictedRating", ascending=False).head(10)
    return predictions_df

def get_movie_poster(movie_id):
    """Fetch movie poster from the base URL."""
    image_url = f"{BASE_URL}{movie_id}.jpg"
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None

# Set custom CSS for hoverable star ratings and layout
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
        .stButton button {
            background-color: orange;
            color: black;
            font-size: 16px;
        }
        .star-rating {
            direction: rtl;
            display: flex;
            justify-content: center;
        }
        .star-rating input {
            display: none;
        }
        .star-rating label {
            font-size: 24px;
            color: gray;
            cursor: pointer;
        }
        .star-rating input:checked ~ label,
        .star-rating label:hover,
        .star-rating label:hover ~ label {
            color: gold;
        }
        .movie-card {
            text-align: center;
            margin: 20px;
        }
        .movie-title {
            font-size: 14px;
            font-weight: bold;
            color: white;
        }
        .movie-genres {
            font-size: 12px;
            color: lightgray;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
movies = load_movies()
rmat = extract_and_load_ratings()

# Streamlit App Layout
st.title("🎥 Movie Recommendation System")
st.markdown("### Step 1: Rate the movies below using stars")

# Display random sample movies for rating
sample_movies = movies.sample(30, random_state=42)  # Increased the sample size to 30
ratings = {}

# Display 6 movies per row
columns_per_row = 6

for start in range(0, len(sample_movies), columns_per_row):
    row_movies = sample_movies.iloc[start : start + columns_per_row]
    columns = st.columns(columns_per_row)
    
    for col, (_, movie) in zip(columns, row_movies.iterrows()):
        movie_id = movie["MovieID"]
        title = movie["Title"]
        genres = movie["Genres"]
        
        # Display poster
        poster = get_movie_poster(movie_id)
        with col:
            if poster:
                st.image(poster, use_container_width=True)
            st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="movie-genres">{genres}</div>', unsafe_allow_html=True)
            
            # Hoverable star-based ratings
            star_rating_html = f"""
            <div class="star-rating">
                {' '.join([f'<input type="radio" id="star-{movie_id}-{5-i}" name="rating-{movie_id}" value="{5-i}"><label for="star-{movie_id}-{5-i}">★</label>' for i in range(5)])}
            </div>
            """
            st.markdown(star_rating_html, unsafe_allow_html=True)


st.markdown("### Step 2: Discover movies you might like")
if st.button("Click here to get your recommendations"):
    new_user = pd.Series(data=np.nan, index=rmat.columns)
    for movie_id, rating in ratings.items():
        new_user[f"m{movie_id}"] = rating

    # Get recommendations using collaborative filtering
    recommendations = myIBCF(new_user)

    if recommendations.empty:
        st.warning("Not enough data to generate recommendations.")
    else:
        st.subheader("🎬 Recommended Movies for You:")
        for idx, movie_id in enumerate(recommendations.index):
            movie_title = movies.loc[movies["MovieID"] == int(movie_id[1:]), "Title"].values[0]
            poster = get_movie_poster(int(movie_id[1:]))
            col1, col2 = st.columns([1, 4])
            if poster:
                col1.image(poster, use_container_width=True)
            col2.markdown(f"**{movie_title}**")

