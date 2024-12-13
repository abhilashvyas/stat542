
import streamlit as st
import pandas as pd
import numpy as np

# Load data function with modified movies.dat loading
def load_data():
    # Load movies.dat from the URL
    movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
    movies = pd.read_csv(movies_url, sep='::', engine='python', encoding="ISO-8859-1", header=None)
    movies.columns = ['MovieID', 'Title', 'Genres']
    
    # Generate image URLs for movies
    movies['image_url'] = movies['MovieID'].apply(lambda x: f"https://liangfgithub.github.io/MovieImages/{x}.jpg?raw=true")

    # Load other required data
    top10_movies = pd.read_csv('movie_popularity.csv')  # Assuming popularity data is precomputed
    similarity_matrix = pd.read_csv('similarity_matrix_top_30.csv')  # Precomputed similarity matrix
    
    return movies, top10_movies, similarity_matrix

# Item-Based Collaborative Filtering function
def myIBCF(newuser, similarity_matrix, movies):
    predictions = {}

    for i in newuser.index:
        if not pd.isna(newuser[i]):
            continue

        # Get similarity values for movie i
        sim_values = similarity_matrix.loc[i]

        # Filter movies with non-NA similarity and rated by the user
        relevant_movies = sim_values[sim_values.notna() & newuser.notna()]
        ratings = newuser[relevant_movies.index]

        # Compute the numerator and denominator for the prediction
        numerator = np.sum(relevant_movies * ratings)
        denominator = np.sum(relevant_movies)

        # Avoid division by zero
        if denominator > 0:
            predictions[i] = numerator / denominator
        else:
            predictions[i] = np.nan

    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['PredictedRating'])
    predictions_df = predictions_df.sort_values(by='PredictedRating', ascending=False)
    top_10 = predictions_df.head(10)

    # If fewer than 10 predictions, fill with popular movies
    if len(top_10) < 10:
        unrated_movies = movies[~movies['MovieID'].isin(newuser[newuser.notna()].index)]
        additional_movies = unrated_movies.head(10 - len(top_10))
        top_10 = pd.concat([top_10, additional_movies])

    return top_10

# Load the datasets
movies, top10_movies, similarity_matrix = load_data()

# Streamlit app starts here
st.title("Movie Recommender System")

st.subheader("Rate a set of sample movies to get personalized recommendations!")

# Display a random set of sample movies for the user to rate
sample_movies = movies.sample(10)
ratings = {}

st.write("### Please rate the following movies:")
for _, row in sample_movies.iterrows():
    rating = st.slider(f"{row['Title']} ({row['Genres']})", min_value=0, max_value=5, step=1, value=0)
    ratings[row['MovieID']] = rating

# Convert user ratings into a Series
user_ratings = pd.Series(np.nan, index=movies['MovieID'])
for movie_id, rating in ratings.items():
    if rating > 0:  # Include only movies that the user has rated
        user_ratings[movie_id] = rating

# Predict recommendations when the user submits
if st.button("Get Recommendations"):
    if user_ratings.notna().sum() == 0:
        st.warning("Please rate at least one movie to get recommendations.")
    else:
        st.write("Generating recommendations...")
        top_10_recommendations = myIBCF(user_ratings, similarity_matrix, movies)

        st.write("### Top 10 Movie Recommendations:")
        for idx, row in top_10_recommendations.iterrows():
            movie_id = idx
            movie_info = movies[movies['MovieID'] == movie_id].iloc[0]
            st.image(movie_info['image_url'], width=150, caption=f"{movie_info['Title']} (Predicted Rating: {row['PredictedRating']:.2f})")

