import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    st.title("Hospital Recommendation System")

    df = pd.read_csv('centre.csv')

    # Ensure the 'Index' column exists
    if 'Index' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Index'}, inplace=True)
    
    selected_features = ['Location', 'Specialization', 'Cost (INR)', 'Ratings']
    st.write("Selected Features:", selected_features)

    for feature in selected_features:
        df[feature] = df[feature].fillna('')

    combined_features = df['Location'] + ' ' + df['Specialization']

    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    similarity = cosine_similarity(feature_vectors)

    name = st.text_input("Enter your addiction among these (Substance Abuse Rehabilitation, Alcohol Rehabilitation, Drug Addiction Rehabilitation, Gambling Addiction Rehabilitation, Smoking Cessation Program): ")
    st.write("Entered addiction:", name)

    list_of_all_titles = df['Specialization'].tolist()
    st.write("List of all titles:", list_of_all_titles)

    find_close_match = difflib.get_close_matches(name, list_of_all_titles)
    st.write("Close match:", find_close_match)
    close_match = find_close_match[0] if find_close_match else None
    st.write("Closest match:", close_match)

    if close_match:
        index_of_the_addiction = df[df.Specialization == close_match]['Index'].values
        st.write("Index of the addiction:", index_of_the_addiction)
        if len(index_of_the_addiction) > 0:
            index_of_the_addiction = int(index_of_the_addiction[0])
            if 0 <= index_of_the_addiction < len(similarity):
                similarity_score = list(enumerate(similarity[index_of_the_addiction]))
                sorted_similar_hospital = sorted(similarity_score, key=lambda x: x[1], reverse=True)
                
                st.write("Sorted similar hospitals:", sorted_similar_hospital)

                # Extract the top 20 indices
                top_indices = [x[0] for x in sorted_similar_hospital[:20]]

                # Print hospitals using the indices
                st.subheader('Rehabilitation centers for your addiction:')
                top_hospitals = df.loc[top_indices]
                for idx, row in top_hospitals.iterrows():
                    st.write(f"{idx + 1}. {row['Name']}, {row['Ratings']}, {row['Location']}")
            else:
                st.warning("Invalid index_of_the_addiction.")
        else:
            st.warning("No data found for the given addiction.")
    else:
        st.warning("No close match found for the given addiction.")

if __name__ == "__main__":
    main()