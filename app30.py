
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

# Set custom CSS for black background and styling
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
        .stMarkdown h1, h2, h3 {
            color: orange;
        }
        .rating-stars input {
            display: none;
        }
        .rating-stars label {
            font-size: 20px;
            color: gray;
            cursor: pointer;
        }
        .rating-stars input:checked ~ label,
        .rating-stars label:hover,
        .rating-stars label:hover ~ label {
            color: gold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
movies = load_movies()
rmat = extract_and_load_ratings()

# Streamlit App Layout
st.markdown("## Step 1: Rate as many movies as possible")
st.markdown("Scroll down to see all 30 movies and rate them using stars!")

# Display random sample movies for rating
sample_movies = movies.sample(30, random_state=42)  # Increased to 30
ratings = {}

# Create a scrollable container for movies
with st.container():
    for _, row in sample_movies.iterrows():
        movie_id = row["MovieID"]
        title = row["Title"]
        genres = row["Genres"]
        
        # Fetch and display movie poster
        col1, col2 = st.columns([1, 4])
        poster = get_movie_poster(movie_id)
        if poster:
            col1.image(poster, use_column_width=True)
        else:
            col1.write("No poster available")
        
        # Display title and genres
        col2.markdown(f"**{title} ({genres})**")
        
        # Add star rating
        col2.markdown(f"""
        <div class="rating-stars">
            {' '.join([f'<input type="radio" id="star{5-i}-{movie_id}" name="rating-{movie_id}" value="{5-i}"><label for="star{5-i}-{movie_id}">★</label>' for i in range(5)])}
        </div>
        """, unsafe_allow_html=True)
        ratings[movie_id] = st.radio(f"Rate {title}", [1, 2, 3, 4, 5], key=f"rating-{movie_id}", index=2)

st.markdown("## Step 2: Discover movies you might like")
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
                col1.image(poster, use_column_width=True)
            col2.markdown(f"**{movie_title}**")
