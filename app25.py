
import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import os

# Load datasets
@st.cache
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

@st.cache
def extract_and_load_ratings(zip_path="Rmat.csv.zip"):

    with zipfile.ZipFile(zip_path, 'r') as z:
        # Extract the CSV file
        csv_name = z.namelist()[0]
        z.extract(csv_name)
    # Load the CSV into a DataFrame
    ratings = pd.read_csv(csv_name, index_col=0)
    # Clean up extracted file
    os.remove(csv_name)
    return ratings

@st.cache
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

# Load data
movies = load_movies()
rmat = extract_and_load_ratings()

# Streamlit App Layout
st.title("Movie Recommendation System")
st.subheader("Rate Movies")

# Display random sample movies for rating
sample_movies = movies.sample(10)
ratings = {}

for _, row in sample_movies.iterrows():
    ratings[row["MovieID"]] = st.radio(
        f"{row['Title']} ({row['Genres']})",
        options=[1, 2, 3, 4, 5],
        index=2,
        key=f"rating-{row['MovieID']}",
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
        st.subheader("Recommended Movies")
        for movie_id in recommendations.index:
            movie_title = movies[movies["MovieID"] == int(movie_id[1:])]["Title"].values[0]
            st.write(f"- {movie_title}")

# Footer
st.sidebar.title("About")
st.sidebar.info(

)

