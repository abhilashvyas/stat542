
import streamlit as st
import pandas as pd
import numpy as np

# Load pre-saved similarity matrix and rating matrix
@st.cache
def load_data():
    file_path = "/content/drive/My Drive/Colab Notebooks/STAT542/Projects/Project 4/App/Rmat.csv"

# Load the CSV file
    R = pd.read_csv(file_path, index_col=0)

    S = pd.read_csv("similarity_matrix_top_30.csv", index_col=0)
    return R, S

R, S = load_data()

# IBCF function
def myIBCF(newuser, S):
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
    return predictions_df.head(10)

# Streamlit UI
st.title("Movie Recommendation System")

# User selects their ID or creates a new profile
user_type = st.radio("Select user type:", ["Existing User", "New User"])

if user_type == "Existing User":
    user_id = st.selectbox("Select your user ID:", R.index)
    user_ratings = R.loc[user_id]
elif user_type == "New User":
    user_ratings = pd.Series(data=np.nan, index=R.columns)
    sample_movies = ["m1", "m10", "m100", "m1510", "m260", "m3212"]
    st.write("Rate the following movies (1-5):")
    for movie in sample_movies:
        user_ratings[movie] = st.slider(f"Rate {movie}", 1, 5, 3)

# Generate Recommendations
if st.button("Get Recommendations"):
    recommendations = myIBCF(user_ratings, S)
    st.subheader("Top 10 Movie Recommendations")
    st.table(recommendations)

# Footer
st.write("Movie Recommendation System built with Streamlit.")

