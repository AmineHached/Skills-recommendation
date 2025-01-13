import pandas as pd
import streamlit as st
import json
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import ast


def main():
    # Load the JSON file for ESB programs
    file_path_esb = 'data/esb.json'
    with open(file_path_esb, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame and process
    esbprograms = pd.DataFrame(data).fillna('normal')
    esbprograms['Métiers'] = esbprograms['Métiers'].apply(lambda x: x if isinstance(x, list) else [])
    esbprograms2 = esbprograms.explode('Métiers').reset_index(drop=True)

    # Load the JSON file for RTMC jobs
    file_path_jobs = 'data/rtmc.json'
    with open(file_path_jobs, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame and process
    jobscrapRTMC = pd.DataFrame(data)
    jobscrapRTMC['appellations'] = jobscrapRTMC['appellations'].apply(lambda x: x if isinstance(x, list) else [])
    jobscrapRTMC2 = jobscrapRTMC.explode('appellations').reset_index(drop=True)
    jobscrapRTMC2.rename(columns={'job_name': 'secteurs', 'appellations': 'job_title'}, inplace=True)
    jobscrapRTMC2['job_title'] = jobscrapRTMC2['job_title'].fillna('').astype(str)
    esbprograms2['Métiers'] = esbprograms2['Métiers'].fillna('').astype(str)

    # Define a function to find the best match for each job title
    def find_best_match(job_title, metier_list):
        match = process.extractOne(job_title, metier_list, scorer=fuzz.token_sort_ratio)
        return match  # Returns (matched_string, score, index)

    # Match job titles
    matches = [
        find_best_match(job_title, esbprograms2['Métiers'])
        for job_title in jobscrapRTMC2['job_title']
    ]
    matches_df = pd.DataFrame(matches, columns=['Matched_Métier', 'Score', 'Index'])
    matches_df = matches_df[matches_df['Score'] >= 60]

    # Merge DataFrames based on matches
    jobscrapRTMC2['Matched_Métier'] = matches_df['Matched_Métier']
    jobscrapRTMC2['Score'] = matches_df['Score']
    jobscrapRTMC2['Index'] = matches_df['Index']
    matched_jobscrapRTMC2 = jobscrapRTMC2.dropna(subset=['Matched_Métier'])
    df = matched_jobscrapRTMC2.merge(
        esbprograms2,
        left_on='Index',
        right_index=True,
        suffixes=('_job', '_program')
    ).drop(columns=['Index', 'Score', 'Matched_Métier'])

    # Load a pre-trained multilingual model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Function to find missed and matched skills using semantic similarity
    def find_missed_and_matched_skills(row):
        skills = ast.literal_eval(row['compétences']) if isinstance(row['compétences'], str) else row['compétences']
        competences = ast.literal_eval(row['Compétences']) if isinstance(row['Compétences'], str) else row['Compétences']

        skills = skills if isinstance(skills, list) else []
        competences = competences if isinstance(competences, list) else []

        matched_skills, missed_skills = [], []

        skills_embeddings = model.encode(skills, convert_to_tensor=True)
        competences_embeddings = model.encode(competences, convert_to_tensor=True)

        for i, skill_embedding in enumerate(skills_embeddings):
            similarity_scores = util.pytorch_cos_sim(skill_embedding, competences_embeddings)[0]
            best_match_idx = similarity_scores.argmax().item()
            best_score = similarity_scores[best_match_idx].item()

            if best_score >= 0.50:
                matched_skills.append(competences[best_match_idx])
            else:
                missed_skills.append(skills[i])

        return pd.Series([missed_skills, matched_skills])

    # Apply the function to find skills
    df[['missed skills', 'matched skills']] = df.apply(find_missed_and_matched_skills, axis=1)

    return df

df = main()  # Call the main function to generate the DataFrame
# Execute the main function
def app():

    # Streamlit application title
    st.title("ESB Skills Recommendations 🤹‍♂️🤹‍♀️")

    # Check if DataFrame is empty
    if df.empty:
        st.error("The DataFrame is empty. Please check your data or processing steps.")
    else:
        # Dropdown for `title` (Program)
        selected_program_title = st.selectbox("Select a Program", df["title"].dropna().unique())

        # Filter the DataFrame based on the selected program title
        program_filtered_df = df[df["title"] == selected_program_title]

        if program_filtered_df.empty:
            st.error("No records found for the selected program.")
        else:
            # Dropdown for `Parcours` (based on filtered program)
            available_parcours = program_filtered_df["Parcours"].dropna().unique()
            selected_parcours = st.selectbox("Select a Parcours", available_parcours)

            # Further filter the DataFrame based on the selected Parcours
            final_filtered_df = program_filtered_df[program_filtered_df["Parcours"] == selected_parcours]

            if final_filtered_df.empty:
                st.error("No records found for the selected parcours.")
            else:
                # Display the filtered results with `job_title` and `missed skills`
                st.write("Filtered Results:")
                st.dataframe(final_filtered_df[['job_title', 'missed skills']],hide_index=True)
