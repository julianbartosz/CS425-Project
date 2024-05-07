import pandas as pd

# Matthew Storm
def getData():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('data/data.csv')

    if df is None:
        raise Exception("Data not found")

    # Map Yes/No values to 1/0 for binary columns
    df['Fever'] = df['Fever'].map({'Yes': 1, 'No': 0})
    df['Cough'] = df['Cough'].map({'Yes': 1, 'No': 0})
    df['Fatigue'] = df['Fatigue'].map({'Yes': 1, 'No': 0})
    df['Difficulty Breathing'] = df['Difficulty Breathing'].map({'Yes': 1, 'No': 0})

    # Map gender to 0/1
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Fahran Sarkar
    # Map categorical levels to integers for Blood Pressure and Cholesterol Level
    df['Blood Pressure'] = df['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
    df['Cholesterol Level'] = df['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})

    # Map Outcome Variable to 1/0
    df['Outcome Variable'] = df['Outcome Variable'].map({'Positive': 1, 'Negative': 0})

    # Fill missing values with column means or defaults
    df.fillna({
        'Fever': df['Fever'].mean(),
        'Cough': df['Cough'].mean(),
        'Fatigue': df['Fatigue'].mean(),
        'Difficulty Breathing': df['Difficulty Breathing'].mean(),
        'Age': df['Age'].mean(),
        'Gender': 1,  # Assume Male as default
        'Blood Pressure': 1,  # Assume Normal as default
        'Cholesterol Level': 1  # Assume Normal as default
    }, inplace=True)

    return df


def preprocess_data(df):
    return getData()

# Julian Bartosz
def calculate_mean_of_group(df):
    df = preprocess_data(df)
    # Calculate the mean of each column grouped by 'Disease'
    grouped = df.groupby('Disease').mean()
    return grouped

# Nick Luedtke
def predict_disease(df, patient_data):
    group_means = calculate_mean_of_group(df)
    patient_series = pd.Series(patient_data)

    # Calculate Euclidean distance between input data and group means
    distances = {}
    for disease, row in group_means.iterrows():
        distances[disease] = ((row - patient_series) ** 2).sum()

    # Find the disease with the smallest distance
    closest_match = min(distances, key=distances.get)
    return closest_match

# Andrew Ly
def get_user_input():
    patient_data = {}
    # Collect user input for various symptoms and demographics
    patient_data['Fever'] = int(input("Do you have a fever? Enter 1 for Yes, 0 for No: "))
    patient_data['Cough'] = int(input("Do you have a cough? Enter 1 for Yes, 0 for No: "))
    patient_data['Fatigue'] = int(input("Do you feel fatigue? Enter 1 for Yes, 0 for No: "))
    patient_data['Difficulty Breathing'] = int(input("Do you have difficulty breathing? Enter 1 for Yes, 0 for No: "))
    patient_data['Age'] = int(input("Enter your age: "))
    patient_data['Gender'] = int(input("Enter your gender: 1 for Male, 0 for Female: "))
    patient_data['Blood Pressure'] = int(input("Enter your blood pressure: 0 for Low, 1 for Normal, 2 for High: "))
    patient_data['Cholesterol Level'] = int(
        input("Enter your cholesterol level: 0 for Low, 1 for Normal, 2 for High: "))
    patient_data['Outcome Variable'] = 0  # Not used for prediction, kept as a placeholder

    return patient_data

# Fahran Sarkar
if __name__ == '__main__':
    df = getData()

    # Get user input for patient data
    patient_data = get_user_input()

    # Predict the disease based on the user input
    predicted_disease = predict_disease(df, patient_data)
    print(f"Predicted Disease: {predicted_disease}")
