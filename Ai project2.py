import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier

# Read the dataset
dataset = pd.read_csv('C:/Users/etc/.spyder-py3/Ai project/Training Dataset.csv')

# Drop 'Loan_ID' column as it is not useful
dataset = dataset.drop('Loan_ID', axis=1)



# Encode the target variable 'Loan_Status'
label_encoder = LabelEncoder()
dataset['Loan_Status'] = label_encoder.fit_transform(dataset['Loan_Status'])  # Y=1, N=0
# Handle categorical variables
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
dataset[categorical_columns] = dataset[categorical_columns].apply(LabelEncoder().fit_transform)



# Handle missing values for all columns
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset = pd.DataFrame(imp_mean.fit_transform(dataset), columns=dataset.columns)
# fit calculate mean , transform replace missing 



# Separate features (X) and target variable (Y)
X = dataset.drop('Loan_Status', axis=1).values  # Features
Y = dataset['Loan_Status'].values  # Target variable

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#random_state=0 ensures that the split is reproducible





# Scale features using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#fit_transform(): Fits the scaler to the training data and scales it.
#transform(): Applies the same scaling to the test data.

# Decision Tree Classifier
DT = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
y_pred_dt = DT.fit(x_train, y_train).predict(x_test)

# Neural Network Classifier
mlp = MLPClassifier(
    random_state=0,
    max_iter=500,
    learning_rate_init=0.001,
    hidden_layer_sizes=(100, 50),
    solver='adam'
)
y_pred_mlp = mlp.fit(x_train, y_train).predict(x_test)

# Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred_nb = gnb.predict(x_test)

# Evaluate model performances
dt_score = accuracy_score(y_pred_dt, y_test)
mlp_score = accuracy_score(y_pred_mlp, y_test)
nb_score = accuracy_score(y_pred_nb, y_test)

# Print accuracy results
print(f"Decision Tree Accuracy: {dt_score}")
print(f"Neural Network Accuracy: {mlp_score}")
print(f"Naive Bayes Accuracy: {nb_score}")

# Plot Decision Tree for visualization
tree.plot_tree(DT)

# Feature importance for Decision Tree
importances = DT.feature_importances_
feature_names = dataset.columns[:-1]  # All columns except the last are features
for i, importance in enumerate(importances):
    print(f"Feature: {feature_names[i]}, Importance: {importance:.4f}")

# Identify features with low importance
threshold = 0.01  # Set a threshold for "low importance" features
low_importance_features = [feature_names[i] for i, imp in enumerate(importances) if imp < threshold]

print("\nLow Importance Features (to drop):")
print(low_importance_features)

# Initialize and train the Decision Tree Classifier again (if necessary)
X = dataset.iloc[:, :-1]  # All columns except the last
y = dataset.iloc[:, -1]   # Last column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

# Get feature importances
importances = DT.feature_importances_
feature_names = X.columns  # Feature column names

# Display feature importances
for i, importance in enumerate(importances):
    print(f"Feature: {feature_names[i]}, Importance: {importance:.4f}")
