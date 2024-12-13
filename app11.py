
import streamlit as st
import pandas as pd
import numpy as np

# Define the myIBCF function
def myIBCF(newuser):
  
    # Load the similarity matrix
    S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)

    # Initialize predictions
    predictions = {}

    # Iterate over all movies in the new user's vector
    for i in newuser.index:
        if not pd.isna(newuser[i]):  # Skip already rated movies
            continue
        if i not in S.index:  # Ensure the movie ID exists in the similarity matrix
            continue
        sim_values = S.loc[i]
        relevant_movies = sim_values[sim_values.notna() & newuser.notna()]
        ratings = newuser[relevant_movies.index]
        numerator = np.sum(relevant_movies * ratings)
        denominator = np.sum(relevant_movies)
        predictions[i] = numerator / denominator if denominator > 0 else np.nan

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['PredictedRating'])
    predictions_df = predictions_df.sort_values(by='PredictedRating', ascending=False)
    return predictions_df.head(10)

# Streamlit App
st.title("Basic Movie Recommender System")
st.write("Provide your ratings to get personalized movie recommendations.")

# Load Rmat.csv
try:
    R = pd.read_csv("Rmat.csv", index_col=0)
except FileNotFoundError:
    st.error("Rmat.csv file not found. Please upload it to proceed.")
    st.stop()

# Select a user from Rmat.csv
user_ids = R.index.tolist()
selected_user = st.selectbox("Select a user ID for testing:", user_ids)

# Display ratings for the selected user
newuser = R.loc[selected_user]
st.write(f"Ratings for user {selected_user}:")
st.write(newuser.dropna())

# Run myIBCF function and show recommendations
if st.button("Get Recommendations"):
    recommendations = myIBCF(newuser)
    if recommendations.empty:
        st.write("No recommendations could be generated. Please try another user or ensure the data is correctly formatted.")
    else:
        st.write("Top 10 Recommended Movies (IDs Only):")
        st.write(recommendations)

# Test a hypothetical user
st.header("Test with a Hypothetical User")
hypothetical_user = pd.Series(data=np.nan, index=R.columns)
hypothetical_user["m1613"] = st.slider("Rate movie m1613:", 0, 5, 0)
hypothetical_user["m1755"] = st.slider("Rate movie m1755:", 0, 5, 0)

if st.button("Get Recommendations for Hypothetical User"):
    recommendations_hypo = myIBCF(hypothetical_user)
    if recommendations_hypo.empty:
        st.write("No recommendations could be generated for the hypothetical user. Please provide valid ratings.")
    else:
        st.write("Top 10 Recommended Movies (IDs Only):")
        st.write(recommendations_hypo)
 movies were rated. Please rate at least one movie.")

