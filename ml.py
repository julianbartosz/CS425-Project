import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("./data.csv")

# plt.figure(figsize=(10, 6))
# sns.histplot(data['Age'], bins=20, kde=True)
# plt.title('Distribution of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# plt.figure(figsize=(10,6))
# sns.countplot(x='Gender',data=data)
# plt.title('count of gender')
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.show()

categorical_features = data.select_dtypes(include=['object']).columns
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

print("\nCategorical Features:", categorical_features)
print("\nNumerical Features:", numerical_features)



label_encoder = LabelEncoder()
for feature in categorical_features:
    data[feature] = label_encoder.fit_transform(data[feature])



X = data.drop('Outcome Variable', axis=1)
y = data['Outcome Variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = tree.DecisionTreeClassifier()

# 10-fold cross validation
scores = cross_val_score(clf, X, y, cv = 10, scoring='f1_macro')
print(scores.mean())

# rf_classifier = RandomForestClassifier()
# rf_classifier.fit(X_train, y_train)

# Predict on test data
# y_pred = rf_classifier.predict(X_test)



# Evaluate accuracy
# accuracy = metrics.accuracy_score(y_test, y_pred)
# print("\nModel Accuracy:", accuracy)


