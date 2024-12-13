
import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Function definition for Item-Based Collaborative Filtering
def myIBCF(newuser):

    # Load the similarity matrix
    S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)

    # Initialize predictions
    predictions = {}

    # Iterate over all movies in the new user's vector
    for i in newuser.index:
        # Skip movies that the user has already rated
        if not pd.isna(newuser[i]):
            continue

        # Get similarity values for movie i
        sim_values = S.loc[i]

        # Filter movies with non-NA similarity and rated by the user
        relevant_movies = sim_values[sim_values.notna() & newuser.notna()]

        # Get ratings for the relevant movies
        ratings = newuser[relevant_movies.index]

        # Compute the numerator and denominator for the prediction
        numerator = np.sum(relevant_movies * ratings)
        denominator = np.sum(relevant_movies)

        # Avoid division by zero
        if denominator > 0:
            predictions[i] = numerator / denominator
        else:
            predictions[i] = np.nan

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['PredictedRating'])

    # Sort by predicted rating in descending order
    predictions_df = predictions_df.sort_values(by='PredictedRating', ascending=False)

    # Select the top 10 movies
    top_10 = predictions_df.head(10)

    # If fewer than 10 predictions, add movies based on popularity
    if len(top_10) < 10:
        # Load the popularity ranking (assumed to be precomputed)
        popularity_ranking = pd.read_csv("movie_popularity.csv", index_col=0)

        # Exclude movies already rated by the user
        unrated_movies = popularity_ranking[~popularity_ranking.index.isin(newuser[newuser.notna()].index)]

        # Fill in the remaining slots with the most popular movies
        additional_movies = unrated_movies.head(10 - len(top_10))
        additional_movies.columns = ['PredictedRating']
        top_10 = pd.concat([top_10, additional_movies])

    return top_10

# Function to fetch movie posters
def fetch_movie_poster(movie_id):
    base_url = "https://liangfgithub.github.io/MovieImages/"
    image_url = f"{base_url}{str(movie_id).zfill(7)}.jpg"
    response = requests.get(image_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None

# Streamlit app
st.title("Movie Recommender System")
st.header("Step 1: Rate as many movies as possible")

# Load and parse movies.dat
movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
movies = pd.read_csv(movies_url, sep="::", engine="python", encoding="ISO-8859-1", header=None)
movies.columns = ['MovieID', 'Title', 'Genres']

# Randomly sample 10 movies for rating
movie_sample = movies.sample(10)
user_ratings = {}

# Display movie posters with rating inputs
for _, row in movie_sample.iterrows():
    movie_id = row['MovieID']
    title = row['Title']
    poster = fetch_movie_poster(movie_id)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if poster:
            st.image(poster, caption=title, width=150)
    with col2:
        rating = st.radio(f"Rate '{title}'", [0, 1, 2, 3, 4, 5], horizontal=True, key=int(movie_id))
        user_ratings[movie_id] = rating

# Step 2: Get Recommendations
st.header("Step 2: Discover movies you might like")
if st.button("Click here to get your recommendations"):
    # Convert user ratings to a Series
    user_ratings_series = pd.Series(user_ratings)
    
    # Get recommendations using the myIBCF function
    recommendations = myIBCF(user_ratings_series)

    # Display the top 10 recommendations with posters
    st.write("Your movie recommendations:")
    for i, (movie_id, row) in enumerate(recommendations.iterrows()):
        title = movies[movies['MovieID'] == movie_id]['Title'].values[0]
        poster = fetch_movie_poster(movie_id)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if poster:
                st.image(poster, caption=f"Rank {i+1}", width=150)
        with col2:
            st.write(f"Rank {i+1}: {title}")


