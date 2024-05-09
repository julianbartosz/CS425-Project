import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# Matthew Storm: Responsible for loading and initial data display
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Data loaded:\n", df.head())
    return df


# Julian Bartosz: Handles data preprocessing and normalization
def preprocess_data(df, is_training_data=True, scaler=None, mean=None):
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

    numeric_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
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
    categorical_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']
    for column in categorical_columns:
        plot_distribution(df, column, f"Distribution of {column}")


# Nick Luedtke: Applies PCA and clustering to the processed data
def perform_pca_and_clustering(df):
    print("Data before PCA and clustering:\n", df.head())
    df_numeric = df.select_dtypes(include=[np.number])
    pca = PCA(n_components=0.95)
    df_pca = pca.fit_transform(df_numeric)
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(df_pca)
    print("Cluster Centers:\n", kmeans.cluster_centers_)
    print("Explained Variance Ratio:\n", pca.explained_variance_ratio_)
    silhouette_avg = silhouette_score(df_pca, kmeans.labels_)
    print("Silhouette Score: ", silhouette_avg)
    return pca, kmeans, kmeans.labels_


def map_clusters_to_diseases(df, labels):
    df['Cluster'] = labels
    cluster_to_disease = df.groupby('Cluster')['Disease'].agg(lambda x: x.mode()[0]).to_dict()
    print("Cluster to disease mapping:\n", cluster_to_disease)
    return cluster_to_disease


# Andrew Ly: Maps disease predictions based on clustering results
def predict_disease(patient_data, pca, kmeans, cluster_to_disease_map, scaler, mean):
    patient_df = pd.DataFrame([patient_data])
    print("Patient data before preprocessing:\n", patient_df.head())
    patient_df, _, _ = preprocess_data(patient_df, is_training_data=False, scaler=scaler, mean=mean)
    print("Patient data after preprocessing:\n", patient_df.head())
    patient_pca = pca.transform(patient_df)
    cluster_label = kmeans.predict(patient_pca)[0]
    return cluster_to_disease_map[cluster_label]


if __name__ == '__main__':
    df = load_data('data/data.csv')
    df_preprocessed, scaler, mean = preprocess_data(df)
    plot_all_distributions(df_preprocessed)  # Plot distributions for each column
    train_df, test_df = train_test_split(df_preprocessed, test_size=0.2, random_state=42)

    train_df_for_pca = train_df.drop(columns=['Disease']).copy() if 'Disease' in train_df.columns else train_df.copy()

    pca, kmeans, labels = perform_pca_and_clustering(train_df_for_pca)
    train_df['Disease'] = df['Disease'].iloc[train_df.index]
    cluster_to_disease_map = map_clusters_to_diseases(train_df, labels)

    test_sample = {
        'Fever': 1, 'Cough': 0, 'Fatigue': 1, 'Difficulty Breathing': 0,
        'Age': 30, 'Gender': 1, 'Blood Pressure': 2, 'Cholesterol Level': 1
    }
    predicted_disease = predict_disease(test_sample, pca, kmeans, cluster_to_disease_map, scaler, mean)
    print(f"Predicted Disease: {predicted_disease}")
