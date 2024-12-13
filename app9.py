
import streamlit as st
import pandas as pd
import numpy as np
import requests

# Function definition for Item-Based Collaborative Filtering
def myIBCF(newuser):
    S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)
    predictions = {}
    for i in newuser.index:
        if not pd.isna(newuser[i]):
            continue
        if i not in S.index:  # Ensure movie ID exists in similarity matrix
            continue
        sim_values = S.loc[i]
        relevant_movies = sim_values[sim_values.notna() & newuser.notna()]
        ratings = newuser[relevant_movies.index]
        numerator = np.sum(relevant_movies * ratings)
        denominator = np.sum(relevant_movies)
        predictions[i] = numerator / denominator if denominator > 0 else np.nan

    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['PredictedRating'])
    predictions_df = predictions_df.sort_values(by='PredictedRating', ascending=False)
    return predictions_df.head(10)

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

# Display rating inputs for sampled movies
for movie_id, row in movie_sample.iterrows():
    title = row['Title']
    rating = st.radio(f"Rate '{title}'", [0, 1, 2, 3, 4, 5], horizontal=True, key=movie_id)
    if rating > 0:
        user_ratings[movie_id] = rating  # Use the correct movie ID format

# Step 2: Get Recommendations
st.header("Step 2: Discover movies you might like")
if st.button("Click here to get your recommendations"):
    # Convert user ratings to a Series
    if user_ratings:
        user_ratings_series = pd.Series(user_ratings, index=user_ratings.keys())
        
        # Debug: Show user ratings
        st.write("User Ratings:", user_ratings_series)

        # Get recommendations using the myIBCF function
        recommendations = myIBCF(user_ratings_series)

        # Display the top 10 recommendations
        if not recommendations.empty:
            st.write("Your movie recommendations:")
            for i, (movie_id, row) in enumerate(recommendations.iterrows(), start=1):
                title = movies.loc[movie_id, 'Title']
                st.write(f"Rank {i}: {title} (Predicted Rating: {row['PredictedRating']:.2f})")
        else:
            st.write("No recommendations could be generated. Please try rating more movies.")
    else:
        st.write("No movies were rated. Please rate at least one movie.")

