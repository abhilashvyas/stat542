
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

    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = z.namelist()[0]
        z.extract(csv_name)
    ratings = pd.read_csv(csv_name, index_col=0)
    os.remove(csv_name)
    return ratings

@st.cache_data
def load_similarity_matrix():

    return pd.read_csv("similarity_matrix_top_30.csv", index_col=0)

# Collaborative Filtering Function
def myIBCF(newuser):

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

    image_url = f"{BASE_URL}{movie_id}.jpg"
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None

# Load data
movies = load_movies()
rmat = extract_and_load_ratings()

# Streamlit App Layout
st.title("🎥 Movie Recommendation System")
st.markdown("### Please rate the following movies to get personalized recommendations:")

# Display random sample movies for rating
sample_movies = movies.sample(30, random_state=42)  # Increase the sample size to 30
ratings = {}

# Adjust layout to handle 30 movies
columns = st.columns(5)  # Create a grid with 5 columns
for idx, row in enumerate(sample_movies.iterrows()):
    col = columns[idx % 5]  # Cycle through columns
    movie_id = row[1]["MovieID"]
    title = row[1]["Title"]
    genres = row[1]["Genres"]

    # Fetch and display movie poster
    poster = get_movie_poster(movie_id)
    if poster:
        col.image(poster, use_column_width=True)
    else:
        col.write("No poster available")

    # Display title and genres
    col.markdown(f"**{title}**")
    col.markdown(f"*{genres}*")

    # Rating input
    ratings[movie_id] = col.radio(
        "Select a rating", [1, 2, 3, 4, 5], key=f"rating-{movie_id}", index=2
    )

# Get user input as a vector
if st.button("Get Recommendations"):
    new_user = pd.Series(data=np.nan, index=rmat.columns)
    for movie_id, rating in ratings.items():
        new_user[f"m{movie_id}"] = rating

    # Get recommendations using collaborative filtering
    recommendations = myIBCF(new_user)

    if recommendations.empty:
        st.warning("Not enough data to generate recommendations.")
    else:
        st.subheader("🎬 Recommended Movies for You:")
        rec_columns = st.columns(5)  # Create a grid for recommendations
        for idx, movie_id in enumerate(recommendations.index):
            col = rec_columns[idx % 5]
            movie_title = movies.loc[movies["MovieID"] == int(movie_id[1:]), "Title"].values[0]
            poster = get_movie_poster(int(movie_id[1:]))
            if poster:
                col.image(poster, use_column_width=True)
            col.markdown(f"**{movie_title}**")

# Footer
st.sidebar.title("About")
st.sidebar.info(

)

