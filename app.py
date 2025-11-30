
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

import random
import math
import streamlit as st
import random
import math

# Add your existing code here
url = "https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/pre-processed-MIMICIII-SA(arx_gen).csv"
df_org_mimic = pd.read_csv(url)
# GS:
#url1 = "https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/Experimental%20Outcome%20MIMIC%20III.csv"
url1 = "experimental_outputs/experimental_outcome_mimic_with_t.xlsx"
#df_exp_mimic = pd.read_csv(url1)
df_exp_mimic = pd.read_excel(url1)

df_res_init = df_org_mimic[['L6_age', 'L2_los_hours', 'L2_Admission_Type', 'L1_Diseases', 'L2_Ethnicity']].copy()
df_org = df_org_mimic[['L7_age', 'L7_los_hours', 'L2_Admission_Type', 'L2_Diseases', 'L3_Ethnicity']].copy()

url2 = "https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/datasets/adult_v1_gen.csv"
df_org_adult = pd.read_csv(url2)

# GS:
#url3 = "https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/Experimental%20Outcome%20Adult.csv"
#df_exp_adult = pd.read_csv(url3)
url3 = "experimental_outputs/experimental_outcome_adult_with_t.xlsx"
df_exp_adult = pd.read_excel(url1)

df_res_init_adult = df_org_adult[['sex', 'L6_age', 'L2_race', 'L3_education', 'L3_occupation', 'L3_workclass', 'L2_marital_status', 'L1_native_country']].copy()


# Define function for the page overview of MIMIC III Dataset
def mimic_dataset_overview():

    def calculate_suppression_fraction(df, k, test_solution):
        # Group the dataset by the quasi-identifiers
        grouped = df.groupby(test_solution)
        suppressed_indices = []

        # Identify groups with less than k rows and mark for suppression
        for group_name, group in grouped:
            if len(group) < k:
                suppressed_indices.extend(group.index)

        # Drop suppressed rows
        df_k = df.drop(suppressed_indices)

        # Restore the original index
        df_index = df_k.index
        df_k = df_k.reset_index(drop=True)

        # Calculate suppression fraction
        suppressed_rows = len(suppressed_indices)
        total_rows = len(df)
        initial_suppression = suppressed_rows / total_rows
        initial_suppression = round(initial_suppression * 100, 2)

        return initial_suppression



    def get_accuracy(df1, k, t, max_suppressed_fraction): # Takes experimental results df as input and returns auroc, solution set and suppression %
        # Filter DataFrame based on k and max_supp values
        #filtered_df = df1[(df1['K'] == k) & (df1['max suppression %'] == max_suppressed_fraction)]
        #GS:
        filtered_df = df1[(df1['K'] == k) & (df1['max suppression %'] == max_suppressed_fraction) & (df1['T'] == t)]

        # Check if any rows match the criteria
        if not filtered_df.empty:
            #GS:
            if pd.isna(filtered_df['solution set SA'].iloc[0]): # in case there no computed solution in the experimental output, to avoid error messages in the app
                #print("It is NULL")
                return None, None, None
            else:
                # Retrieve accuracy value (assuming there's only one match)
                accuracy_value = filtered_df['median auroc'].iloc[0]
                suprression_value = filtered_df['suppression in %'].iloc[0]
                solution_set = filtered_df['solution set SA'].iloc[0]
                solution_set = [item.strip("'") for item in solution_set.split(",")]
                # Clean up the solution set
                solution_set = [col.strip().replace("'", "").replace('"', '') for col in solution_set]

            return accuracy_value, suprression_value, solution_set
        else:
            return None, None, None  # Return None if no matching rows found

    def display_datasets(df_res_init, df_org, subset_df):
        st.header("Display Datasets")
        # Create columns to display datasets side by side
        selected_datasets = st.multiselect("Select datasets to display:", ["Original Dataset", "Initial Dataset", "Optimized Dataset"], default=["Original Dataset"] )
        # Calculate the number of columns needed based on the number of selected datasets
        num_columns = len(selected_datasets)
        # Create columns to display datasets side by side
        cols = st.columns(num_columns)


        for i, dataset in enumerate(selected_datasets):
            if dataset == "Original Dataset":
                with cols[i]:
                    st.write("### Original Dataset")
                    st.dataframe(df_org)
            elif dataset == "Initial Dataset":
                with cols[i]:
                    st.write("### Initial Dataset")
                    st.dataframe(df_res_init)
            elif dataset == "Optimized Dataset":
                with cols[i]:
                    st.write("### Optimized Dataset")
                    st.dataframe(subset_df)



    def display_datasets_with_distribution(original_dataset, anon_dataset):
          st.subheader("Display change in data distribution after anonymization:")
          tab1, tab2, tab3, tab4, tab5 = st.tabs(["Age", "Gender", "Ethnicity", "Diseases", "Death within 360 days"])

          with tab1:
              # Filter columns containing the word "age"
              age_column = [col for col in anon_dataset.columns if 'age' in col.lower()]
              print("Age column?", age_column)

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["L7_age"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Age")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Age (Original Dataset)")

              sns.histplot(anon_dataset[age_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Age")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Age (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab2:
              # Filter columns containing the word "gender"
              gender_column = [col for col in anon_dataset.columns if 'gender' in col.lower()]
              #print("Age column?", gender_column)

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["gender"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Gender")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of gender (Original Dataset)")

              sns.histplot(anon_dataset[gender_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Gender")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Gender (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab3:
              # Filter columns containing the word "ethnicity"
              ethnicity_column = [col for col in anon_dataset.columns if 'ethnicity' in col.lower()]
              #print("Age column?", ethnicity_column)

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["L3_Ethnicity"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Ethnicity")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Ethnicity (Original Dataset)")

              sns.histplot(anon_dataset[ethnicity_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("L3_Ethnicity")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Ethnicity (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab4:
              # Filter columns containing the word "diseases"
              diseases_column = [col for col in anon_dataset.columns if 'diseases' in col.lower()]
              #print("Age column?", diseases_column)

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["diseases"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Diseases")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Diseases (Original Dataset)")

              sns.histplot(anon_dataset[diseases_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Diseases")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Diseases (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab5:
              # Filter columns containing the word "death"
              death_column = [col for col in anon_dataset.columns if 'death' in col.lower()]

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["death_365_days"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Diseases")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Diseases (Original Dataset)")

              sns.histplot(anon_dataset[death_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Death within 365 days")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Death within 365 days (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

    # Define the Streamlit app title
    st.header("Simulated Annealing Optimization - MIMIC III")
    st.markdown("MIMIC-III integrates deidentified, comprehensive clinical data of patients admitted to the Beth Israel Deaconess Medical Center in Boston, Massachusetts it is preprocessed to predict 365-day mortality probability among intensive care unit (ICU) stroke inpatients between 2001 and 2012.")
    #st.markdown(''' :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in] :gray[pretty] :rainbow[colors].''')
    st.markdown(''':green[Predictors:] age, length of stay/h, admission type, gender, stroke type, marital status, diseases, ethnicity, infarct type, anion gap, abp diastolic, abp systolic, bicarbonate, calcium, chloride, creatinine, gcs, glucose, heart rate, hemoglobin,  oasis, wbc, platelet count, potassium, respiratory rate, sodium ''')
    # Quasi Identifiers
    #st.markdown("<span style='color:green'>**Quasi Identifiers:**</span>", unsafe_allow_html=True)
    st.markdown( ''':green[Quasi identifiers:] age, length of stay/h, admission type, gender, stroke type, marital status, diseases, ethnicity ''')

    ##################################################--------------------SIDEBAR-------------------#######################################
    # Create Streamlit widgets for user input
    st.sidebar.header("User Settings")
    k = st.sidebar.slider("k-Anonymity Threshold", min_value=2, max_value=10, value=2, step=1)

    max_suppressed_fraction = st.sidebar.slider("Maximum suppression", min_value=10, max_value=30, value=20, step=10)

    #GS:
    t = st.sidebar.slider("t-Closeness Threshold", min_value=0.6, max_value=0.7, value=0.6, step=0.1)

    accuracy_value, suprression_value, solution_set = get_accuracy(df_exp_mimic, k, t, max_suppressed_fraction)

    initial_solution = ['L6_age', 'L2_los_hours', 'L2_Admission_Type', 'gender', 'stroke_type', 'marital_status', 'L1_Diseases', 'L2_Ethnicity']
    #k_init = st.sidebar.slider("k-Anonymity on initial solution", min_value=2, max_value=10, value=2, step=1)
    initial_suppression = calculate_suppression_fraction(df_org_mimic, k, initial_solution) # change to percentage to have suppression percentage of initial random solution when k is applied




    ##################################################--------------------Privacy Indicator (Ampel)-------------------#######################################
    def categorize_solution(solution_set):
        # Extract numbers from column names
        numbers = [int(''.join(filter(str.isdigit, col))) for col in solution_set if any(char.isdigit() for char in col)]

        # Sum up the numbers
        total = sum(numbers)

        # Normalize the score
        #normalized_score = (total - min(numbers)) / (max(numbers) - min(numbers))

       # Reverse the score
        #reversed_score = 1 - normalized_score

      # Categorize based on the reversed normalized score
        if 5 <= total <= 10:
            return total
        elif 11 <= total <= 15:
            return total
        else:
            return total

    # Example usage:
    #solution_set = ['L1_age', 'L2_length', 'L3_height', 'L4_width', 'L5_weight']
    if solution_set is not None: # GS: added to execute this only if there is calculated solution
        score = categorize_solution(solution_set)
        #print("Categorized Result:", categorized_result)  # Output: "Good"
        #print("Score:", score)  # Output: The total sum of numbers
        score_init = categorize_solution(df_res_init.columns)

        privacy_indicator_value = score




    ##################################################--------------------Initial vs Best Solution and Optimize BTN and METRIC-------------------#######################################

    col1, col2 = st.columns(2)

    with col1:
      # Display the initial solution
      st.write("Initial Solution:", initial_solution)
      #st.write("Classification score AUROC:", '74.5')

    if st.sidebar.button("Start Optimization"):
      with col2:
        # GS: add notification for the case, when there are no calculated solutions
        if solution_set is None or accuracy_value is None or suprression_value is None:
            st.error("⚠️ No solution found for the given criteria (k, t, and suppression limit). Please try adjusting the parameters.")
        else:
            # Inside the if condition, display the optimization results
            st.write("Best Solution:", solution_set)


        #st.write("Suppressed Rows:", suprression_value)
        #st.write("Best solution AUROC", accuracy_value)
        #st.write("Optimized dataset vs. Original dataset")

      # GS: added check so that the results are displayed when the solution set is computed
      if solution_set is not None and accuracy_value is not None and suprression_value is not None:
          st.subheader("Comparing Datasets before and after Anonymization")
          tab1, tab2 = st.tabs(["Original Dataset", "Initial Random Dataset"])
          with tab1:
                st.markdown("**Anonymized Dataset vs Original Dataset**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Percentage of suppressed datapoints",
                        value=str(suprression_value) + "%",  # Format as percentage with 2 decimal points
                        delta=str(round(suprression_value - 40, 2)) + "%"
                    )

                with col2:
                    st.metric(
                        label="Classification score (AUROC)",
                        value=accuracy_value,
                        delta=round(accuracy_value - 0.77, 3)
                    )

                with col3:
                    # Define the privacy value with the appropriate emoji
                    privacy_value_with_emoji = "😊" if privacy_indicator_value <= 5 else ("😐" if 6 <= privacy_indicator_value <= 15 else "😔")

                    # Concatenate the privacy value and emoji with a space
                    privacy_info = f"{privacy_value_with_emoji}"

                    # Display the metric with the privacy info
                    st.metric(
                        label="Data Privacy",
                        value=privacy_info,
                        delta=privacy_indicator_value - 22  # You can set delta to None if you don't need it
                    )
                    with st.popover("💡"):
                        st.markdown("The higher the data privacy indicator value the greater the risk of data exposure as the data is more granular and less generalized. It ranges from 4 to 22 depending on the sum of generalization levels of all features in the solution set.")


          with tab2:
                st.markdown("**Anonymized Dataset vs. Initial Random Dataset**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Percentage of suppressed datapoints",
                        value=str(suprression_value) + "%",
                        delta=str(round(suprression_value - initial_suppression, 2)) + "%"
                    )

                with col2:
                    st.metric(
                        label="Classification score (AUROC)",
                        value=accuracy_value,
                        delta=round(accuracy_value - 0.745, 3)
                    )

                with col3:
                    # Define the privacy value with the appropriate emoji
                    privacy_value_with_emoji = "😊" if privacy_indicator_value <= 5 else ("😐" if 6 <= privacy_indicator_value <= 15 else "😔")

                    # Concatenate the privacy value and emoji with a space
                    privacy_info = f"{privacy_value_with_emoji} "

                    # Display the metric with the privacy info
                    st.metric(
                        label="Privacy Indicator",
                        value=privacy_info,
                        delta=privacy_indicator_value - score_init  # You can set delta to None if you don't need it
                    )
                    with st.popover("💡"):
                        st.markdown("The higher the data privacy indicator value the greater the risk of data exposure as the data is more granular and less generalized. It ranges from 4 to 22 depending on the sum of generalization levels of all features in the solution set.")
    #col1, col2, col3 = st.sidebar.columns(3)

    ##################################################--------------------Display distribution imbalance after anonymization-------------------#######################################

          subset_df = df_org_mimic[solution_set]
          # Add the 'death_365_days' column from df_org_mimic to the subset_df
          subset_df['death_365_days'] = df_org_mimic['death_365_days']
          # Print the size of the DataFrame
          print("Size of subset_df:", subset_df.shape)
          # Group the dataset by the quasi-identifiers
          grouped = subset_df.groupby(solution_set)
          suppressed_indices = []

          # Identify groups with less than k rows and mark for suppression
          for group_name, group in grouped:
              if len(group) < k:
                  suppressed_indices.extend(group.index)

              # Drop suppressed rows
          subset_df_anon = subset_df.drop(suppressed_indices)
          print("Size of subset_df_anon:", subset_df_anon.shape)

          # Restore the original index
          subset_df_anon = subset_df_anon.reset_index(drop=True)
          print("Size of subset_df_anon after reset index:", subset_df_anon.shape)

          display_datasets(df_res_init, df_org, subset_df_anon)
          display_datasets_with_distribution(df_org_mimic, subset_df_anon)


######################################START OF ADULT CENSUS DATASET OVERVIEW######################################
def adult_census_datset_overview():

    def calculate_suppression_fraction(df, k, t, test_solution):
        # Group the dataset by the quasi-identifiers
        grouped = df.groupby(test_solution)
        suppressed_indices = []

        # Identify groups with less than k rows and mark for suppression
        for group_name, group in grouped:
            if len(group) < k:
                suppressed_indices.extend(group.index)

        # Drop suppressed rows
        df_k = df.drop(suppressed_indices)

        # Restore the original index
        df_index = df_k.index
        df_k = df_k.reset_index(drop=True)

        # Calculate suppression fraction
        suppressed_rows = len(suppressed_indices)
        total_rows = len(df)
        initial_suppression = suppressed_rows / total_rows
        initial_suppression = round(initial_suppression * 100, 2)

        return initial_suppression


    def get_accuracy(df1, k, t, max_suppressed_fraction): # Takes experimental results df as input and returns auroc, solution set and suppression %
        # Filter DataFrame based on k and max_supp values
        #filtered_df = df1[(df1['K'] == k) & (df1['max suppression %'] == max_suppressed_fraction)]
        #GS:
        filtered_df = df1[(df1['K'] == k) & (df1['max suppression %'] == max_suppressed_fraction) & (df1['T'] == t)]

        # Check if any rows match the criteria
        if not filtered_df.empty:
            #GS: the same as for MIMIC
            if pd.isna(filtered_df['solution set SA'].iloc[0]):
                #print("It is NULL")
                return None, None, None
            else:
                # Retrieve accuracy value (assuming there's only one match)
                accuracy_value = filtered_df['median auroc'].iloc[0]
                suprression_value = filtered_df['suppression in %'].iloc[0]
                suprression_value = suprression_value * 100
                solution_set = filtered_df['solution set SA'].iloc[0]
                solution_set = [item.strip("'") for item in solution_set.split(",")]
                # Clean up the solution set
                solution_set = [col.strip().replace("'", "").replace('"', '') for col in solution_set]

                return accuracy_value, suprression_value, solution_set
        else:
            return None, None, None  # Return None if no matching rows found


    def display_datasets(df_res_init, df_org, subset_df):
       st.header("Display Datasets")
       # Create columns to display datasets side by side
       selected_datasets = st.multiselect("Select datasets to display:", ["Original Dataset", "Initial Dataset", "Optimized Dataset"], default=["Original Dataset"] )
       # Calculate the number of columns needed based on the number of selected datasets
       num_columns = len(selected_datasets)
       # Create columns to display datasets side by side
       cols = st.columns(num_columns)


       for i, dataset in enumerate(selected_datasets):
           if dataset == "Original Dataset":
               with cols[i]:
                   st.write("### Original Dataset")
                   st.dataframe(df_org)
           elif dataset == "Initial Dataset":
               with cols[i]:
                   st.write("### Initial Dataset")
                   st.dataframe(df_res_init_adult)
           elif dataset == "Optimized Dataset":
               with cols[i]:
                   st.write("### Optimized Dataset")
                   st.dataframe(subset_df)



    def display_datasets_with_distribution(original_dataset, anon_dataset):
          st.subheader("Display change in data distribution after anonymization:")
          tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Age", "Sex", "Education", "Occupation", "Marital status", "Death within 360 days"])

          with tab1:
              # Filter columns containing the word "age"
              age_column = [col for col in anon_dataset.columns if 'age' in col.lower()]
              print("Age column?", age_column)

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["L7_age"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Age")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Age (Original Dataset)")
              axes[0].tick_params(axis='x', labelrotation=90)

              sns.histplot(anon_dataset[age_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Age")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Age (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab2:
              # Filter columns containing the word "Sex"
              sex_column = [col for col in anon_dataset.columns if 'sex' in col.lower()]

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["sex"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Sex")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of sex (Original Dataset)")
              #axes[0].tick_params(axis='x', labelrotation=90)

              sns.histplot(anon_dataset[sex_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Sex")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Sex (Anonymized Dataset)")
              #axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab3:
              # Filter columns containing the word "Education"
              education_column = [col for col in anon_dataset.columns if 'education' in col.lower()]
              #print("Age column?", education_column)

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["L5_education"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Education")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Education (Original Dataset)")
              axes[0].tick_params(axis='x', labelrotation=90)

              sns.histplot(anon_dataset[education_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("L3_education")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Education (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab4:
              # Filter columns containing the word "Occupation"
              occupation_column = [col for col in anon_dataset.columns if 'occupation' in col.lower()]
              #print("Age column?", occupation_column)

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["L3_occupation"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Occupation")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Occupation (Original Dataset)")
              axes[0].tick_params(axis='x', labelrotation=90)

              sns.histplot(anon_dataset[occupation_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Occupation")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Occupation (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab5:
              # Filter columns containing the word "Marital status"
              marital_status_column = [col for col in anon_dataset.columns if 'marital_status' in col.lower()]

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["L3_marital_status"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Marital Status")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Marital status (Original Dataset)")
              axes[0].tick_params(axis='x', labelrotation=90)

              sns.histplot(anon_dataset[marital_status_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Marital status")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Marital status (Anonymized Dataset)")
              axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)

          with tab6:
              # Filter columns containing the word "Salary"
              salary_class_column = [col for col in anon_dataset.columns if 'salary-class' in col.lower()]

              fig, axes = plt.subplots(1, 2, figsize=(12, 5))
              sns.histplot(original_dataset["salary-class"], kde=True, color="blue", label="Original Dataset", ax=axes[0])
              axes[0].set_xlabel("Salary")
              axes[0].set_ylabel("Frequency")
              axes[0].set_title("Distribution of Salary (Original Dataset)")

              sns.histplot(anon_dataset[salary_class_column[0]], kde=True, color="orange", label="Anonymized Dataset", ax=axes[1])
              axes[1].set_xlabel("Salary")
              axes[1].set_ylabel("Frequency")
              axes[1].set_title("Distribution of Salary (Anonymized Dataset)")
              #axes[1].tick_params(axis='x', labelrotation=90)

              plt.tight_layout()
              st.pyplot(fig)



    # Define the Streamlit app title
    st.header("Simulated Annealing Optimization - Adult Census")
    st.markdown("The Adult Census dataset, contains demographic information about individuals, such as age, education, marital status, occupation, and income level. It is commonly used for predictive modeling tasks, particularly for binary classification problems to predict whether an individual earns more than $50,000 per year based on their demographic attributes. ")
    #st.markdown(''' :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in] :gray[pretty] :rainbow[colors].''')
    st.markdown(''':green[Predictors:] age, sex, race, education, occupation, workclass, marital_status, native country ''')
    # Quasi Identifiers
    #st.markdown("<span style='color:green'>**Quasi Identifiers:**</span>", unsafe_allow_html=True)
    st.markdown( ''':green[Quasi identifiers:] age, sex, race, education, occupation, workclass, marital_status, native country ''')

    ##################################################--------------------SIDEBAR-------------------#######################################
    # Create Streamlit widgets for user input
    st.sidebar.header("User Settings")
    k = st.sidebar.slider("k-Anonymity Threshold", min_value=2, max_value=10, value=2, step=1)

    max_suppressed_fraction = st.sidebar.slider("Maximum suppression", min_value=10, max_value=30, value=20, step=10)
    #GS:
    t = st.sidebar.slider("t-Closeness Threshold", min_value=0.6, max_value=0.7, value=0.6, step=0.1)

    accuracy_value, suprression_value, solution_set = get_accuracy(df_exp_adult, k, t, max_suppressed_fraction)

    initial_solution = ['sex', 'L6_age', 'L2_race', 'L3_education', 'L3_occupation', 'L3_workclass', 'L2_marital_status', 'L1_native_country']
    #k_init = st.sidebar.slider("k-Anonymity on initial solution", min_value=2, max_value=10, value=2, step=1)
    initial_suppression = calculate_suppression_fraction(df_org_adult, k, initial_solution) # change to percentage to have suppression percentage of initial random solution when k is applied

    ##################################################--------------------Privacy Indicator (Ampel)-------------------#######################################
    def categorize_solution(solution_set):
        # Extract numbers from column names
        numbers = [int(''.join(filter(str.isdigit, col))) for col in solution_set if any(char.isdigit() for char in col)]

        # Sum up the numbers
        total = sum(numbers)

        # Normalize the score
        #normalized_score = (total - min(numbers)) / (max(numbers) - min(numbers))

       # Reverse the score
        #reversed_score = 1 - normalized_score

      # Categorize based on the reversed normalized score
        if 7 <= total <= 12:
            return total
        elif 13 <= total <= 19:
            return total
        else:
            return total

    # Example usage:
    if solution_set is not None: # GS: the same as for MIMIC
        score = categorize_solution(solution_set)
        #print("Categorized Result:", categorized_result)  # Output: "Good"
        #print("Score:", score)  # Output: The total sum of numbers
        score_init = categorize_solution(df_res_init_adult.columns)


        privacy_indicator_value = score


     ##################################################--------------------Initial vs Best Solution and Optimize BTN and METRIC-------------------#######################################

    col1, col2 = st.columns(2)

    with col1:
      # Display the initial solution
      st.write("Initial Solution:", initial_solution)
      #st.write("Classification score AUROC:", '74.5')

    if st.sidebar.button("Start Optimization"):
      with col2:
        # GS: the same notification as for MIMIC
        if solution_set is None or accuracy_value is None or suprression_value is None:
            st.error("⚠️ No solution found for the given criteria (k, t, and suppression limit). Please try adjusting the parameters.")
        else:
            # Inside the if condition, display the optimization results
            st.write("Best Solution:", solution_set)


        #st.write("Suppressed Rows:", suprression_value)
        #st.write("Best solution AUROC", accuracy_value)
        #st.write("Optimized dataset vs. Original dataset")
      # GS: the same notification as for MIMIC
      if solution_set is not None and accuracy_value is not None and suprression_value is not None:
          st.subheader("Comparing Datasets before and after Anonymization")
          tab1, tab2 = st.tabs(["Original Dataset", "Initial Random Dataset"])
          with tab1:
                st.markdown("**Anonymized Dataset vs Original Dataset**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Percentage of suppressed datapoints",
                        value=str(suprression_value) + "%",  # Format as percentage with 2 decimal points
                        delta=str(round(suprression_value - 50, 2)) + "%"
                    )

                with col2:
                    st.metric(
                        label="Classification score (AUROC)",
                        value=accuracy_value,
                        delta=round(accuracy_value - 0.77, 3)
                    )

                with col3:
                    # Define the privacy value with the appropriate emoji
                    privacy_value_with_emoji = "😊" if privacy_indicator_value <= 12 else ("😐" if 13 <= privacy_indicator_value <= 19 else "😔")

                    # Concatenate the privacy value and emoji with a space
                    privacy_info = f"{privacy_value_with_emoji} "

                    # Display the metric with the privacy info
                    st.metric(
                        label="Privacy Indicator",
                        value=privacy_info,
                        delta=privacy_indicator_value - 22  # You can set delta to None if you don't need it
                    )
                    with st.popover("💡"):
                        st.markdown("The higher the data privacy indicator value the greater the risk of data exposure as the data is more granular and less generalized. It ranges from 7 to 26 depending on the sum of generalization levels of all features in the solution set.")


          with tab2:
                st.markdown("**Anonymized Dataset vs. Initial Random Dataset**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Percentage of suppressed datapoints",
                        value=str(suprression_value) + "%",
                        delta=str(round(suprression_value - initial_suppression, 2)) + "%"
                    )

                with col2:
                    st.metric(
                        label="Classification score (AUROC)",
                        value=accuracy_value,
                        delta=round(accuracy_value - 0.732, 3)
                    )

                with col3:
                    # Define the privacy value with the appropriate emoji
                    privacy_value_with_emoji = "😊" if privacy_indicator_value <= 12 else ("😐" if 13 <= privacy_indicator_value <= 19 else "😔")

                    # Concatenate the privacy value and emoji with a space
                    privacy_info = f"{privacy_value_with_emoji} "

                    # Display the metric with the privacy info
                    st.metric(
                        label="Privacy Indicator",
                        value=privacy_info,
                        delta=privacy_indicator_value - score_init  # You can set delta to None if you don't need it
                    )
                    with st.popover("💡"):
                        st.markdown("The higher the data privacy indicator value the greater the risk of data exposure as the data is more granular and less generalized. It ranges from 7 to 26 depending on the sum of generalization levels of all features in the solution set.")

   ##################################################--------------------Display distribution imbalance after anonymization-------------------#######################################

          subset_df = df_org_adult[solution_set]
          # Add the 'salary-class' column from df_org_adult to the subset_df
          subset_df['salary-class'] = df_org_adult['salary-class']
          # Print the size of the DataFrame
          print("Size of subset_df:", subset_df.shape)
          # Group the dataset by the quasi-identifiers
          grouped = subset_df.groupby(solution_set)
          suppressed_indices = []

          # Identify groups with less than k rows and mark for suppression
          for group_name, group in grouped:
              if len(group) < k:
                  suppressed_indices.extend(group.index)

              # Drop suppressed rows
          subset_df_anon = subset_df.drop(suppressed_indices)
          print("Size of subset_df_anon:", subset_df_anon.shape)

          # Restore the original index
          subset_df_anon = subset_df_anon.reset_index(drop=True)
          print("Size of subset_df_anon after reset index:", subset_df_anon.shape)

    #st.subheader("Original Dataset")
    #st.dataframe(df_org)

      # Function to display datasets
          display_datasets(df_res_init_adult, df_org_adult, subset_df_anon)
          display_datasets_with_distribution(df_org_adult, subset_df_anon)



#####################################################HOME PAGE################################################
# Define function for home page
def home():
    st.title("PrivacyShade Demo App")
    st.subheader("Welcome to the Utility Preserving Data Anonymization Demo App")
    st.write("*The idea is based on **Simulated Annealing**, an optimization technique inspired by the metallurgical annealing process. It is essential in solving the challenge of searching hierarchical data generalizations. It begins by exploring various configurations, adjusting them iteratively while probabilistically accepting worse solutions early to thoroughly explore the solution space. This ensures that both your privacy requirements, such as **k-anonymity** constraints, and your goal of achieving **high classification scores** are effectively balanced. As the algorithm progresses, it refines its approach, aiming to find the optimal solution where privacy is preserved without compromising data utility.*")
    #st.write("Select a dataset to explore:")
    selected_datasets = st.selectbox("Select a dataset to explore:", ["MIMIC III Dataset", "Adult Income Dataset"])
    if selected_datasets == "MIMIC III Dataset":
          st.markdown("**:green[Dataset description:]** 360-day mortality prediction among intensive care unit (ICU) stroke inpatients")
          #st.markdown(''' :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in] :gray[pretty] :rainbow[colors].''')
          st.markdown('''**:green[Predictors:]** age, length of stay/h, admission type, gender, stroke type, marital status, diseases, ethnicity, infarct type, anion gap, abp diastolic, abp systolic, bicarbonate, calcium, chloride, creatinine, gcs, glucose, heart rate, hemoglobin,  oasis, wbc, platelet count, potassium, respiratory rate, sodium ''')
          # Quasi Identifiers
          #st.markdown("<span style='color:green'>**Quasi Identifiers:**</span>", unsafe_allow_html=True)
          st.markdown( ''' **:green[Quasi identifiers:]** age, length of stay/h, admission type, gender, stroke type, marital status, diseases, ethnicity ''')
          st.markdown(" **:green[Dataset size:]** 2655 rows and 27 columns")
          st.subheader("Dataset Overview")
          selected_columns = ['L7_age', 'gender', 'L3_Ethnicity', 'marital_status', 'infarct_type',
          'anion_gap', 'abp_diastolic', 'abp_systolic', 'bicarbonate', 'calcium',
          'chloride', 'creatinine', 'gcs', 'glucose', 'heart_rate', 'hemoglobin',
          'oasis', 'L7_los_hours', 'platelet_count', 'potassium',
          'repiratory_rate', 'sodium', 'wbc', 'stroke_type', 'L2_Admission_Type', 'diseases', 'death_365_days']
          st.dataframe(df_org_mimic[selected_columns])
          st.subheader("MIMIC III Generalization Hierarchies")
          tab1, tab2, tab3, tab4 = st.tabs(["Age", "Ethnicity", "Diseases", "Admission Type"])
          with tab1:
            #st.header("Age")
            st.image("https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/age_mimic.png")
          with tab2:
            #st.header("Admission Type")
            st.image("https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/diseases_mimic.png")
          with tab3:
            #st.header("Ethnicity")
            st.image("https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/ethnicity_mimic.png")
          with tab4:
            st.image("https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/admissin_mimic.png")

    if selected_datasets == "Adult Income Dataset":
          st.markdown("**:green[Dataset description:]** The Adult Census dataset, contains demographic information about individuals, such as age, education, marital status, occupation, and income level. It is commonly used for predictive modeling tasks, particularly for binary classification problems to predict whether an individual earns more than $50,000 per year based on their demographic attributes.")
          #st.markdown(''' :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in] :gray[pretty] :rainbow[colors].''')
          st.markdown('''**:green[Predictors:]** age, sex, race, education, occupation, workclass, marital_status, native country ''')
          # Quasi Identifiers
          #st.markdown("<span style='color:green'>**Quasi Identifiers:**</span>", unsafe_allow_html=True)
          st.markdown( ''' **:green[Quasi identifiers:]** age, sex, race, education, occupation, workclass, marital_status, native country ''')
          st.markdown(" **:green[Dataset size:]** 32,561 rows and 9 columns")
          st.subheader("Dataset Overview")

          st.dataframe(df_org_adult)
          st.subheader("Adult Census Generalization Hierarchies")
          tab1, tab2, tab3, = st.tabs(["Education", "Occupation", "Marital Status"])
          with tab1:
            #st.header("Age")
            st.image("https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/education_adult.png")
          with tab2:
            #st.header("Education")
            st.image("https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/occupation_adult.png")
          with tab3:
            #st.header("Occupation")
            st.image("https://raw.githubusercontent.com/desstaw/PrivacyPreservingTechniques/main/SA%20Demo%20App/marital_status_adult.png")



# Define function for MIMIC III Dataset overview
def mimic_overview():
    mimic_dataset_overview()

# Define function for Adult Census Dataset overview
def adult_census_overview():
    st.title("Overview of Adult Census Dataset")
    # Add content for Adult Census Dataset overview
    adult_census_datset_overview()





# Main function to run the app
def main():
    # Create sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Home", "Optimizing MIMIC III Dataset", "Optimizing Adult Census Dataset"])

    # Display different pages based on selection
    if page == "Home":
        home()
    elif page == "Optimizing MIMIC III Dataset":
        mimic_overview()
    elif page == "Optimizing Adult Census Dataset":
        adult_census_overview()



if __name__ == "__main__":
    main()
