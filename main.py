import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyinputplus as pyip
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DEBUG_MODE = False

# Matthew Storm: Responsible for loading and initial data display
def load_data(filepath):
    df = pd.read_csv(filepath)
    if DEBUG_MODE:
        print("Data loaded:\n", df.head())
    return df


# Julian Bartosz: Handles data preprocessing and normalization
def preprocess_data(df, is_training_data=True, scaler=None, mean=None):
    if DEBUG_MODE:
        print("Data before preprocessing:\n", df.head())
    mappings = {
        'Fever': {'Yes': 1, 'No': 0},
        'Cough': {'Yes': 1, 'No': 0},
        'Fatigue': {'Yes': 1, 'No': 0},
        'Difficulty Breathing': {'Yes': 1, 'No': 0},
        'Gender': {'Female': 0, 'Male': 1},
        'Blood Pressure': {'Low': 0, 'Normal': 1, 'High': 2},
        'Cholesterol Level': {'Low': 0, 'Normal': 1, 'High': 2}
    }
    for col, map_dict in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(map_dict)

    numeric_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure',
                    'Cholesterol Level']
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    if numeric_cols:
        if is_training_data:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            mean = df[numeric_cols].mean()
        else:
            df[numeric_cols] = df[numeric_cols].fillna(mean)

    if is_training_data:
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    if DEBUG_MODE:
        print("Data after preprocessing:\n", df.head())
    return df, scaler, mean


# Farhan Sarkar: Visualizes data distribution for key variables
def plot_distribution(df, column, title):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=df)
    plt.title(title)
    plt.savefig(f"{column}_distribution.png")
    plt.close()


def plot_all_distributions(df):
    categorical_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure',
                           'Cholesterol Level']
    for column in categorical_columns:
        plot_distribution(df, column, f"Distribution of {column}")


# Nick Luedtke: Applies PCA and clustering to the processed data
def perform_pca_and_clustering(df):
    if DEBUG_MODE:
        print("Data before PCA and clustering:\n", df.head())
    df_numeric = df.select_dtypes(include=[np.number])
    pca = PCA(n_components=0.95)
    df_pca = pca.fit_transform(df_numeric)
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df_pca)

    if DEBUG_MODE:
        print("Cluster Centers:\n", kmeans.cluster_centers_)

    if DEBUG_MODE:
        print("Explained Variance Ratio:\n", pca.explained_variance_ratio_)

    silhouette_avg = silhouette_score(df_pca, kmeans.labels_)

    if DEBUG_MODE:
        print("Silhouette Score: ", silhouette_avg)

    return pca, kmeans, kmeans.labels_


def map_clusters_to_diseases(df, labels):
    df['Cluster'] = labels
    cluster_to_disease = df.groupby('Cluster')['Disease'].agg(lambda x: x.mode()[0]).to_dict()
    if DEBUG_MODE:
        print("Cluster to disease mapping:\n", cluster_to_disease)
    return cluster_to_disease


# Andrew Ly: Maps disease predictions based on clustering results
def predict_disease(patient_data, pca, kmeans, cluster_to_disease_map, scaler, mean):
    patient_df = pd.DataFrame([patient_data])
    if DEBUG_MODE:
        print("Patient data before preprocessing:\n", patient_df.head())
    patient_df, _, _ = preprocess_data(patient_df, is_training_data=False, scaler=scaler, mean=mean)
    if DEBUG_MODE:
        print("Patient data after preprocessing:\n", patient_df.head())
    patient_pca = pca.transform(patient_df)
    cluster_label = kmeans.predict(patient_pca)[0]
    return cluster_to_disease_map[cluster_label]


def user_menu(pca, kmeans, cluster_to_disease_map, scaler, mean):
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    print("Welcome to the Disease Prediction System!")

    option = pyip.inputMenu(['Predict Disease', 'View List of Disease\'s', 'Exit'], numbered=True)
    while option != 'Exit':
        print(option)
        if option == "Predict Disease":

            print("Please enter the following information:")
            fever = {"yes": 1, "no": 0}[pyip.inputYesNo("Fever? (Yes/No) ")]
            cough = {"yes": 1, "no": 0}[pyip.inputYesNo("Cough? (Yes/No) ")]
            fatigue = {"yes": 1, "no": 0}[pyip.inputYesNo("Fatigue? (Yes/No) ")]
            difficulty_breathing = {"yes": 1, "no": 0}[pyip.inputYesNo("Difficulty Breathing? (Yes/No) ")]
            age = pyip.inputInt("Age? ")
            gender = {"Female": 0, "Male": 1}[pyip.inputMenu(["Female", "Male"], "Gender? \n", numbered=True)]
            cholesterol_level = pyip.inputMenu(['Low', 'Normal', 'High'], "Cholesterol Level: \n", numbered=True)
            blood_pressure = pyip.inputMenu(['Low', 'Normal', 'High'], "Blood Pressure: \n", numbered=True)

            # Map values to numerical values
            cholesterol_level = {'Low': 0, 'Normal': 1, 'High': 2}[cholesterol_level]
            blood_pressure = {'Low': 0, 'Normal': 1, 'High': 2}[blood_pressure]

            patient_data = {
                'Fever': fever, 'Cough': cough, 'Fatigue': fatigue, 'Difficulty Breathing': difficulty_breathing,
                'Age': age, 'Gender': gender, 'Blood Pressure': blood_pressure, 'Cholesterol Level': cholesterol_level
            }

            predicted_disease = predict_disease(patient_data, pca, kmeans, cluster_to_disease_map, scaler, mean)

            print(f"\n\nYou may have: {predicted_disease}\n\n")

            print("Press Key to Continue")

        if option == "View List of Disease\'s":
            print()
            # Use the cluster map to display
            for key in cluster_to_disease_map:
                print(f"{cluster_to_disease_map[key]}")
            print("\n\n")

        option = pyip.inputMenu(['Predict Disease', 'View List of Disease\'s', 'Exit'], numbered=True)
    pass


if __name__ == '__main__':
    df = load_data('data/data.csv')
    df_preprocessed, scaler, mean = preprocess_data(df)
    plot_all_distributions(df_preprocessed)  # Plot distributions for each column
    train_df, test_df = train_test_split(df_preprocessed, test_size=0.2, random_state=42)

    train_df_for_pca = train_df.drop(columns=['Disease']).copy() if 'Disease' in train_df.columns else train_df.copy()

    pca, kmeans, labels = perform_pca_and_clustering(train_df_for_pca)
    train_df['Disease'] = df['Disease'].iloc[train_df.index]
    cluster_to_disease_map = map_clusters_to_diseases(train_df, labels)

    user_menu(pca, kmeans, cluster_to_disease_map, scaler, mean)
