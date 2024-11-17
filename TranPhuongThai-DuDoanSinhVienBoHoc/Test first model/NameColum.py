# Data Preprocessing (same as in the model)
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def preprocess_data(data):
    # Rename columns for consistency
    data.rename(columns={"Nacionality": "Nationality", 
                         "Mother's qualification": "Mother_qualification", 
                         "Father's qualification": "Father_qualification", 
                         "Mother's occupation": "Mother_occupation", 
                         "Father's occupation": "Father_occupation", 
                         "Age at enrollment": "Age"}, inplace=True)

    # Replace spaces with underscores and remove special characters
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('(', '').str.replace(')', '')

    # Convert categorical columns to 'category' dtype
    cat_columns = ['Marital_status', 'Application_mode', 'Application_order', 
                   'Course', 'Daytime/evening_attendance', 'Previous_qualification', 
                   'Nationality', 'Mother_qualification', 'Father_qualification', 
                   'Mother_occupation', 'Father_occupation', 'Displaced', 
                   'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date', 
                   'Gender', 'Scholarship_holder', 'International', 'Target']
    data[cat_columns] = data[cat_columns].astype('category')

    # Encode 'Target' column (Dropout=0, Enrolled=1, Graduate=2)
    data['Target_encoded'] = OrdinalEncoder(categories=[['Dropout', 'Enrolled', 'Graduate']]).fit_transform(data[['Target']])
    
    # Drop the original 'Target' column
    data.drop('Target', axis=1, inplace=True)

    # Feature Engineering: Calculate averages for the two semesters
    data['avg_credited'] = data[['Curricular_units_1st_sem_credited', 
                                  'Curricular_units_2nd_sem_credited']].mean(axis=1)
    data['avg_enrolled'] = data[['Curricular_units_1st_sem_enrolled', 
                                  'Curricular_units_2nd_sem_enrolled']].mean(axis=1)
    data['avg_evaluations'] = data[['Curricular_units_1st_sem_evaluations', 
                                    'Curricular_units_2nd_sem_evaluations']].mean(axis=1)
    data['avg_approved'] = data[['Curricular_units_1st_sem_approved', 
                                 'Curricular_units_2nd_sem_approved']].mean(axis=1)
    data['avg_grade'] = data[['Curricular_units_1st_sem_grade', 
                              'Curricular_units_2nd_sem_grade']].mean(axis=1)
    data['avg_without_evaluations'] = data[['Curricular_units_1st_sem_without_evaluations', 
                                             'Curricular_units_2nd_sem_without_evaluations']].mean(axis=1)
    
    # Drop the original semester columns since they are now aggregated
    data = data.drop(columns=['Curricular_units_1st_sem_credited', 
                              'Curricular_units_1st_sem_enrolled', 
                              'Curricular_units_1st_sem_evaluations', 
                              'Curricular_units_1st_sem_approved', 
                              'Curricular_units_1st_sem_grade', 
                              'Curricular_units_1st_sem_without_evaluations', 
                              'Curricular_units_2nd_sem_credited', 
                              'Curricular_units_2nd_sem_enrolled', 
                              'Curricular_units_2nd_sem_evaluations', 
                              'Curricular_units_2nd_sem_approved', 
                              'Curricular_units_2nd_sem_grade', 
                              'Curricular_units_2nd_sem_without_evaluations'])

    return data

# Example for web app input (after reading the data):
input_data = pd.read_csv('your_input_file.csv')  # Read the input file (user data)
processed_data = preprocess_data(input_data)  # Preprocess the input data

# Now, you can use 'processed_data' for prediction
