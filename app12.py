
import streamlit as st
import pandas as pd
import numpy as np

# Load the similarity matrix and rating matrix
S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)
R = pd.read_csv("Rmat.csv", index_col=0)

# Define the myIBCF function
def myIBCF(newuser):

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

# Streamlit App UI
st.title("Movie Recommender System")
st.header("Rate Some Movies to Get Recommendations")

# Randomly sample 10 movies for rating
sample_movies = R.columns.to_series().sample(10).tolist()
user_ratings = {}

# Display sliders for sampled movies
for movie_id in sample_movies:
    rating = st.slider(f"Rate '{movie_id}'", min_value=0, max_value=5, step=1, key=movie_id)
    if rating > 0:  # Only add movies with a rating
        user_ratings[movie_id] = rating

# Step 2: Generate Recommendations
if st.button("Get Recommendations"):
    if user_ratings:
        # Convert user ratings to a Series
        user_ratings_series = pd.Series(user_ratings, index=R.columns).fillna(np.nan)

        # Call the myIBCF function
        recommendations = myIBCF(user_ratings_series)

        # Display the recommendations
        if not recommendations.empty:
            st.header("Top 10 Movie Recommendations")
            st.write(recommendations)
        else:
            st.write("No recommendations could be generated. Try rating more movies.")
    else:
        st.write("Please rate at least one movie to get recommendations.")

