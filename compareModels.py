import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import precision_score, recall_score, accuracy_score

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# One Hot Encoding Data
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

# Adding missing features to test data
missing_features = set(train_data.columns) - set(test_data.columns)

for feat in missing_features:
    test_data[feat] = False

test_data = test_data[['pclass_is2nd', 'pclass_is3rd', 'pclass_iscrew', 
                     'age_ischild', 'gender_ismale', 'survived']]

# Separating features and targets
X_train = train_data.drop(columns=['survived'], axis=1)
y_train = train_data['survived']

X_test = test_data.drop(columns=['survived'], axis=1)
y_test = test_data['survived']

# Training and Evaluating Models

### Decision Tree
dct_clf = DecisionTreeClassifier(criterion="entropy")
dct_clf.fit(X_train, y_train)

dct_y_pred = dct_clf.predict(X_test)

dct_precision = precision_score(y_test, dct_y_pred)
dct_recall = recall_score(y_test, dct_y_pred)
dct_accuracy = accuracy_score(y_test, dct_y_pred)

print("Decision Tree Classifier Evaluation Metrics: ")
print("Precision: {:.5f}".format(dct_precision))
print("Recall: {:.5f}".format(dct_recall))
print("Accuracy: {:.5f}".format(dct_accuracy))
print("--------------------------------------------\n")

### Multinomial Naive Bayes Classifier
mnb_clf = MultinomialNB()
mnb_clf.fit(X_train, y_train)

mnb_y_pred = mnb_clf.predict(X_test)

mnb_precision = precision_score(y_test, mnb_y_pred)
mnb_recall = recall_score(y_test, mnb_y_pred)
mnb_accuracy = accuracy_score(y_test, mnb_y_pred)

print("Multinomial Naive Bayes Classifier Evaluation Metrics: ")
print("Precision: {:.5f}".format(mnb_precision))
print("Recall: {:.5f}".format(mnb_recall))
print("Accuracy: {:.5f}".format(mnb_accuracy))
print("--------------------------------------------\n")

### Bernoulli Naive Bayes Classifier
bnb_clf = BernoulliNB()
bnb_clf.fit(X_train, y_train)

bnb_y_pred = bnb_clf.predict(X_test)

bnb_precision = precision_score(y_test, bnb_y_pred)
bnb_recall = recall_score(y_test, bnb_y_pred)
bnb_accuracy = accuracy_score(y_test, bnb_y_pred)

print("Bernoulli Naive Bayes Classifier Evaluation Metrics:")
print("Precision: {:.5f}".format(bnb_precision))
print("Recall: {:.5f}".format(bnb_recall))
print("Accuracy: {:.5f}".format(bnb_accuracy))
print("--------------------------------------------\n")