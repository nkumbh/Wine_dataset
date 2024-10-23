# Wine_dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
                'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
                'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
wine_data = pd.read_csv(url, names=column_names)
wine_data.head()
wine_data.info()
wine_data.describe()
# Step 1.2: Check for duplicates and missing values
print("Number of duplicate rows:", wine_data.duplicated().sum())
wine_data = wine_data.drop_duplicates()

print("Missing values:\n", wine_data.isnull().sum())
# Step 1.3: Class balance
class_counts = wine_data['Class'].value_counts()
print("Class distribution:\n", class_counts)
To determine whether the dataset has balanced or imbalanced classes, we can look at the distribution of class labels. 
In this case, the class distribution is as follows:
Class 1: 59 instances
Class 2: 71 instances
Class 3: 48 instances
The dataset is considered imbalanced when the class distribution is significantly skewed towards one or more classes. 
In this dataset, while the class counts are not perfectly equal, they are not significantly skewed towards any particular class. 
Therefore, we can consider this dataset to have relatively balanced classes.

# Step 1.4: Features correlation
correlation_matrix = wine_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Features Correlation Heatmap")
plt.show()

Looking at the correlation matrix:

Features such as "flavanoids" and "total_phenols" have high positive correlation coefficients (0.86).
Similarly, "flavanoids" and "od280/od315_of_diluted_wines" have high positive correlation coefficients (0.79).
High positive correlation indicates redundancy between these features. In such cases, performing dimensionality reduction techniques like PCA can be beneficial to reduce the number of features while retaining most of the information.

Therefore, based on the high correlation between certain features in the dataset, feature reduction techniques like PCA can be considered to reduce redundancy and improve the efficiency of the model.



# Step 1.5: Feature scaling
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Step 2: Split the dataset into 80% train and 20% test and apply PCA for dimension reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
# Step 3: Use SVC to predict classes and evaluate model performance on test data
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train_pca, y_train)

y_pred = svc.predict(X_test_pca)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
