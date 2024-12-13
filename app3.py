
import streamlit as st
import pandas as pd
import numpy as np

# Load the similarity matrix and rating matrix
S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)
R = pd.read_csv("Rmat.csv", index_col=0)

# Load the movie titles
movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
movies = pd.read_csv(movies_url, sep='::', engine='python', encoding="ISO-8859-1", header=None)
movies.columns = ['MovieID', 'Title', 'Genres']
movies['MovieID'] = movies['MovieID'].apply(lambda x: f"m{x}")  # Prefix MovieID with 'm' to match Rmat columns
movies.set_index('MovieID', inplace=True)

# Define the IBCF function
def myIBCF(newuser):
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
    predictions_df = predictions_df.sort_values(by='PredictedRating', ascending=False)
    top_10 = predictions_df.head(10)

    if len(top_10) < 10:
        # Add unrated movies based on their titles
        unrated_movies = movies[~movies.index.isin(newuser[newuser.notna()].index)]
        additional_movies = unrated_movies.head(10 - len(top_10))
        additional_movies['PredictedRating'] = np.nan
        top_10 = pd.concat([top_10, additional_movies])

    # Join with movie titles for display
    top_10 = top_10.join(movies[['Title']], how='left')
    return top_10[['Title', 'PredictedRating']]

# Streamlit UI
st.title("Movie Recommender System")
st.write("Rate a few movies to get personalized recommendations!")

# Sample movies for the user to rate
sample_movies = ["m1", "m10", "m100", "m1510", "m260", "m3212"]
newuser = pd.Series(data=np.nan, index=R.columns)

st.subheader("Rate the following movies (1-5):")
for movie in sample_movies:
    movie_title = movies.loc[movie, 'Title']
    rating = st.slider(f"Rate '{movie_title}':", 0, 5, 0, key=movie)
    if rating > 0:
        newuser[movie] = rating

if st.button("Get Recommendations"):
    recommendations = myIBCF(newuser)
    st.subheader("Top 10 Recommended Movies:")
    st.table(recommendations)


