
import streamlit as st
import pandas as pd
import numpy as np
import requests

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

    return predictions_df.head(10)

# Streamlit app
st.title("Movie Recommender System")
st.header("Step 1: Rate as many movies as possible")

# Load movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
response = requests.get(myurl)
movie_lines = response.text.split('
')
movie_data = [line.split("::") for line in movie_lines if line]
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

# Randomly sample 10 movies for rating
movie_sample = movies.sample(10)
user_ratings = {}

# Display rating inputs
for _, row in movie_sample.iterrows():
    title = row['title']
    movie_id = row['movie_id']
    rating = st.slider(f"Rate '{title}'", 0, 5, 0, key=movie_id)
    if rating > 0:
        user_ratings[f"m{movie_id}"] = rating  # Prefix with "m" to match similarity matrix

# Step 2: Get Recommendations
st.header("Step 2: Discover movies you might like")
if st.button("Click here to get your recommendations"):
    # Convert user ratings to a Series
    user_ratings_series = pd.Series(user_ratings, index=list(user_ratings.keys()))

    if user_ratings_series.empty:
        st.write("No movies were rated. Please rate at least one movie.")
    else:
        # Get recommendations using the myIBCF function
        recommendations = myIBCF(user_ratings_series)

        # Display the top 10 recommendations
        st.write("Your movie recommendations:")
        for i, (movie_id, row) in enumerate(recommendations.iterrows(), start=1):
            movie_title = movies.loc[movies['movie_id'] == int(movie_id[1:]), 'title'].values[0]
            st.write(f"Rank {i}: {movie_title} (Predicted Rating: {row['PredictedRating']:.2f})")

