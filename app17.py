
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# Load the movies dataset
movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
movies = pd.read_csv(movies_url, sep='::', engine='python', encoding="ISO-8859-1", header=None)
movies.columns = ['MovieID', 'Title', 'Genres']

# Extract and load the rating matrix from the zip file
zip_file_path = "Rmat.csv.zip"
extracted_file_name = "Rmat.csv"

# Check if the file is already extracted to avoid re-extraction
if not os.path.exists(extracted_file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extract(extracted_file_name)

# Load the extracted Rmat.csv file
rmat = pd.read_csv(extracted_file_name, index_col=0)
R = rmat.values
movie_ids = rmat.columns  # Movie IDs corresponding to columns

# Define the Item-Based Collaborative Filtering function
def myIBCF(newuser):
    S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)
    predictions = {}

    for i in newuser.index:
        if not pd.isna(newuser[i]):
            continue

        sim_values = S.loc[i]
        relevant_movies = sim_values[sim_values.notna() & newuser.notna()]
        ratings = newuser[relevant_movies.index]

        numerator = np.sum(relevant_movies * ratings)
        denominator = np.sum(relevant_movies)

        if denominator > 0:
            predictions[i] = numerator / denominator
        else:
            predictions[i] = np.nan

    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['PredictedRating'])
    predictions_df = predictions_df.sort_values(by='PredictedRating', ascending=False)
    top_10 = predictions_df.head(10)

    if len(top_10) < 10:
        popularity_ranking = pd.read_csv("movie_popularity.csv", index_col=0)
        unrated_movies = popularity_ranking[~popularity_ranking.index.isin(newuser[newuser.notna()].index)]
        additional_movies = unrated_movies.head(10 - len(top_10))
        additional_movies.columns = ['PredictedRating']
        top_10 = pd.concat([top_10, additional_movies])

    return top_10

# Streamlit app starts here
st.title("Movie Recommender System")

st.subheader("Rate a set of sample movies to get personalized recommendations!")

# Display a random set of sample movies for the user to rate
sample_movies = movies.sample(10)
ratings = {}

st.write("### Please rate the following movies:")
for idx, row in sample_movies.iterrows():
    rating = st.slider(f"{row['Title']} ({row['Genres']})", min_value=0, max_value=5, step=1, value=0)
    ratings[row['MovieID']] = rating

# Convert user ratings into a Series
user_ratings = pd.Series(np.nan, index=rmat.columns)  # Use exact column names from Rmat.csv
for movie_id, rating in ratings.items():
    column_name = f"m{movie_id}"  # Ensure correct column name format
    if column_name in user_ratings.index and rating > 0:  # Check column existence
        user_ratings[column_name] = rating


# Predict recommendations when the user submits
if st.button("Get Recommendations"):
    if user_ratings.notna().sum() == 0:
        st.warning("Please rate at least one movie to get recommendations.")
    else:
        st.write("Generating recommendations...")
        top_10_recommendations = myIBCF(user_ratings)

        st.write("### Top 10 Movie Recommendations:")
        for idx, row in top_10_recommendations.iterrows():
            movie_id = idx[1:]  # Remove the 'm' prefix
            movie_title = movies.loc[movies['MovieID'] == int(movie_id), 'Title'].values[0]
            st.write(f"{movie_title} (Predicted Rating: {row['PredictedRating']:.2f})")

