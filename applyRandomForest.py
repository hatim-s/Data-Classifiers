import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

train_data = pd.get_dummies(train_data, 
                            columns=['pclass', 'age', 'gender', 'survived'], 
                            drop_first=True)
test_data = pd.get_dummies(test_data, 
                            columns=['pclass', 'age', 'gender', 'survived'],
                            drop_first=True)

train_data.columns = ['pclass_is2nd', 'pclass_is3rd', 'pclass_iscrew', 
                     'age_ischild', 'gender_ismale', 'survived']

test_data.columns = ['pclass_is2nd', 'pclass_is3rd', 
                     'age_ischild', 'gender_ismale', 'survived']

# Adding Missing Features in test data
missing_features = set(train_data.columns) - set(test_data.columns)
for feat in missing_features:
    test_data[feat] = False

test_data = test_data[['pclass_is2nd', 'pclass_is3rd', 'pclass_iscrew', 
                     'age_ischild', 'gender_ismale', 'survived']]

# Preparing training and test features and labels
X_train = train_data.drop(columns=['survived'], axis=1)
y_train = train_data['survived']

X_test = test_data.drop(columns=['survived'], axis=1)
y_test = test_data['survived']

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Fit the model to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy  (Initial Model): {:.5f}%".format(accuracy * 100))
print("Precision (Initial Model): {:.5f}%".format(precision * 100))
print("Recall    (Initial Model): {:.5f}%".format(recall * 100))
print("---------------------------------------------------\n")

# Define a parameter grid to search through
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 1, 2, 3],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [1, 2, 4],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_rf_classifier = grid_search.best_estimator_

# Make predictions with the best model
y_pred_tuned = best_rf_classifier.predict(X_test)

print (best_rf_classifier)

# Evaluate the tuned model's performance
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)

print("Accuracy  (Tuned Model): {:.5f}%".format(accuracy_tuned * 100))
print("Precision (Tuned Model): {:.5f}%".format(precision_tuned * 100))
print("Recall    (Tuned Model): {:.5f}%".format(recall_tuned * 100))
print("---------------------------------------------------\n")