
import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Function definition for Item-Based Collaborative Filtering
def myIBCF(newuser):
    # Load similarity matrix
    S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)
    predictions = {}
    
    for i in newuser.index:
        if not pd.isna(newuser[i]):  # Skip already rated movies
            continue
        if i not in S.index:  # Ensure the movie exists in the similarity matrix
            continue
        sim_values = S.loc[i]
        relevant_movies = sim_values[sim_values.notna() & newuser.notna()]
        ratings = newuser[relevant_movies.index]
        numerator = np.sum(relevant_movies * ratings)
        denominator = np.sum(relevant_movies)
        predictions[i] = numerator / denominator if denominator > 0 else np.nan

    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['PredictedRating'])
    predictions_df = predictions_df.sort_values(by='PredictedRating', ascending=False)

    # Debugging output
    st.write("Predictions DataFrame:", predictions_df)

    return predictions_df.head(10)

# Function to fetch movie posters
def fetch_movie_poster(movie_id):
    try:
        base_url = "https://liangfgithub.github.io/MovieImages/"
        image_url = f"{base_url}{str(movie_id).zfill(7)}.jpg"
        response = requests.get(image_url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception as e:
        st.write(f"Error fetching poster for movie ID {movie_id}: {e}")
    return None

# Streamlit app
st.title("Movie Recommender System")
st.header("Step 1: Rate as many movies as possible")

# Load and parse movies.dat
movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
movies = pd.read_csv(movies_url, sep="::", engine="python", encoding="ISO-8859-1", header=None)
movies.columns = ['MovieID', 'Title', 'Genres']
movies['MovieID'] = movies['MovieID'].apply(lambda x: f"m{x}")  # Prefix MovieID with 'm' to match the similarity matrix
movies.set_index('MovieID', inplace=True)

# Randomly sample 10 movies for rating
movie_sample = movies.sample(10)
user_ratings = {}

# Display movie posters with rating inputs
for movie_id, row in movie_sample.iterrows():
    title = row['Title']
    poster = fetch_movie_poster(movie_id[1:])  # Remove the "m" prefix for the poster fetch

    col1, col2 = st.columns([1, 3])
    with col1:
        if poster:
            st.image(poster, caption=title, width=150)
    with col2:
        rating = st.radio(f"Rate '{title}'", [0, 1, 2, 3, 4, 5], horizontal=True, key=movie_id)
        if rating > 0:
            user_ratings[movie_id] = rating

# Step 2: Get Recommendations
st.header("Step 2: Discover movies you might like")
if st.button("Click here to get your recommendations"):
    # Convert user ratings to a Series
    user_ratings_series = pd.Series(user_ratings, index=list(user_ratings.keys()))

    # Debugging output
    st.write("User Ratings Series:", user_ratings_series)

    if user_ratings_series.empty:
        st.write("No movies were rated. Please rate at least one movie.")
    else:
        # Get recommendations using the myIBCF function
        recommendations = myIBCF(user_ratings_series)

        # Display the top 10 recommendations with posters
        st.write("Your movie recommendations:")
        for i, (movie_id, row) in enumerate(recommendations.iterrows()):
            if movie_id in movies.index:
                title = movies.loc[movie_id, 'Title']
                poster = fetch_movie_poster(movie_id[1:])  # Remove the "m" prefix for the poster fetch

                col1, col2 = st.columns([1, 3])
                with col1:
                    if poster:
                        st.image(poster, caption=f"Rank {i+1}", width=150)
                with col2:
                    st.write(f"Rank {i+1}: {title} (Predicted Rating: {row['PredictedRating']:.2f})")
            else:
                st.write(f"Movie ID {movie_id} not found in movies data.")


