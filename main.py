import pandas as pd


def getData():
    # Read from the data set csv file
    df = pd.read_csv('data.csv')

    if df is None:
        # Throw an error
        raise Exception("Data not found")
    
    df.info()
    # Data Layout
    # Disease, Fever, Cough, Fatigue, Difficulty Breathing, Age, Gender, Blood Pressure, Cholesterol Level, Outcome Variable
    # String, Yes/No, Yes/No, Yes/No, Yes/No, Int, String, String (Low, Normal, High), String (Low, Normal, High), String (Positive, Negative)
    # String, Boolean, Boolean, Boolean, Boolean, Integer, Integer(0-1), Integer(0-2), Boolean

    # Convert the data to Booleans or Integers
    df['Fever'] = df['Fever'].map({'Yes': 1, 'No': 0})
    df['Cough'] = df['Cough'].map({'Yes': 1, 'No': 0})
    df['Fatigue'] = df['Fatigue'].map({'Yes': 1, 'No': 0})
    df['Difficulty Breathing'] = df['Difficulty Breathing'].map({'Yes': 1, 'No': 0})
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Blood Pressure'] = df['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
    df['Cholesterol Level'] = df['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})
    df['Outcome Variable'] = df['Outcome Variable'].map({'Positive': 1, 'Negative': 0})

    # Fill in missing data except for the outcome variable and Disease

    df['Fever'].fillna(df['Fever'].mean(), inplace=True)
    df['Cough'].fillna(df['Cough'].mean(), inplace=True)
    df['Fatigue'].fillna(df['Fatigue'].mean(), inplace=True)
    df['Difficulty Breathing'].fillna(df['Difficulty Breathing'].mean(), inplace=True)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df["Gender"].fillna("Male", inplace=True)
    df["Blood Pressure"].fillna("Normal", inplace=True)
    df["Cholesterol Level"].fillna("Normal", inplace=True)

    # Return the data
    return df


def clusterData():
    pass


def calculateMeanOfGroup():
    pass


def userInput():
    pass

