
import streamlit as st
import pandas as pd
import numpy as np

# Load necessary data
S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)
R = pd.read_csv("R.csv", index_col=0)

# Function to display movies for users to rate
def get_sample_movies(num_movies=100):
    sample_movies = R.columns[:num_movies]
    return sample_movies

# Simplified version of myIBCF function for 100 movies
def myIBCF_limited(newuser, S_limited):
    predictions = pd.Series(index=S_limited.index, dtype=np.float64)

    for i in S_limited.index:
        # Skip movies the user has already rated
        if not pd.isna(newuser[i]):
            predictions[i] = np.nan
            continue

        # Get similarities and ratings for movies rated by the user
        similar_movies = S_limited.loc[i]
        rated_movies = ~newuser.isna()
        
        relevant_similarities = similar_movies[rated_movies]
        relevant_ratings = newuser[rated_movies]

        numerator = np.nansum(relevant_similarities * relevant_ratings)
        denominator = np.nansum(relevant_similarities)

        predictions[i] = numerator / denominator if denominator > 0 else np.nan

    # Sort predictions and return top 10
    return predictions.sort_values(ascending=False).head(10).dropna()

# Streamlit app interface
st.title("Movie Recommender System")

st.write("### Rate a set of movies to get personalized recommendations!")

# Get a sample of 100 movies
sample_movies = get_sample_movies()

# User ratings input
new_user_ratings = {}
st.write("#### Please rate the following movies (leave blank if not seen):")

for movie in sample_movies:
    rating = st.selectbox(f"{movie}", options=[None, 1, 2, 3, 4, 5], format_func=lambda x: "" if x is None else x)
    new_user_ratings[movie] = rating

# Convert user ratings to a Series
new_user_ratings_series = pd.Series(new_user_ratings, index=sample_movies)

# When user submits, process ratings and make recommendations
if st.button("Get Recommendations"):
    # Filter similarity matrix for the 100 displayed movies
    S_limited = S.loc[sample_movies, sample_movies]

    # Get recommendations
    recommendations = myIBCF_limited(new_user_ratings_series, S_limited)

    if recommendations.empty:
        st.write("We couldn't generate enough recommendations based on your ratings. Please rate more movies!")
    else:
        st.write("### Your Top 10 Movie Recommendations:")
        st.table(recommendations)
