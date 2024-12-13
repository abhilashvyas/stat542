
import streamlit as st
import pandas as pd
import numpy as np

# Function definition
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

# Streamlit app
st.title("Movie Recommender System")
st.write("Rate a few movies to get personalized recommendations!")

# Load and parse movies.dat
movies = pd.read_csv(
    "movies.dat",
    sep="::",
    engine="python",
    names=["MovieID", "Title", "Genres"],
    encoding="ISO-8859-1"  # Specify Latin-1 encoding to handle special characters
)

# Randomly sample 10 movies for rating
movie_sample = movies.sample(10)

# User ratings input
user_ratings = {}
st.write("Rate these movies from 1 to 5 (leave blank if you haven't watched):")
for index, row in movie_sample.iterrows():
    movie_id = row['MovieID']
    movie_name = row['Title']
    rating = st.number_input(f"{movie_name}", min_value=1, max_value=5, step=1, key=int(movie_id))
    user_ratings[movie_id] = rating if rating > 0 else np.nan

# Convert user ratings to a Series
user_ratings_series = pd.Series(user_ratings)

# Get recommendations on submit
if st.button("Get Recommendations"):
    recommendations = myIBCF(user_ratings_series)
    st.write("Top 10 Recommendations for You:")
    st.table(recommendations)

