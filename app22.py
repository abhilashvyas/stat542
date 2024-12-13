
import streamlit as st
import pandas as pd
import numpy as np

# Load datasets
@st.cache
def load_data():
    movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
    movies = pd.read_csv(movies_url, sep='::', engine='python', encoding="ISO-8859-1", header=None)
    movies.columns = ['MovieID', 'Title', 'Genres']
    rmat = pd.read_csv("Rmat.csv", index_col=0)
    return movies, rmat

# IBCF function
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
        predictions[i] = numerator / denominator if denominator > 0 else np.nan
    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['PredictedRating'])
    predictions_df = predictions_df.sort_values(by='PredictedRating', ascending=False).head(10)
    return predictions_df

# Streamlit app
st.title("Movie Recommendation System")

# Load data
movies, rmat = load_data()

# Display sample movies for user to rate
st.subheader("Rate Movies")
sample_movies = movies.sample(10)
ratings = {}

for _, row in sample_movies.iterrows():
    ratings[row['MovieID']] = st.slider(f"{row['Title']} ({row['Genres']})", 1, 5, 3)

# Convert user ratings to a Series
newuser = pd.Series(data=np.nan, index=rmat.columns)
for movie_id, rating in ratings.items():
    newuser[f"m{movie_id}"] = rating

# Generate recommendations
if st.button("Get Recommendations"):
    recommendations = myIBCF(newuser)
    if recommendations.empty:
        st.write("Not enough data for recommendations.")
    else:
        st.subheader("Recommended Movies")
        for movie_id in recommendations.index:
            movie_title = movies[movies['MovieID'] == int(movie_id[1:])]['Title'].values[0]
            st.write(f"- {movie_title}")

