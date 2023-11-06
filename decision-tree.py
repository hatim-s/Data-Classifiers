import numpy
import pandas
import math

train_df = pandas.read_csv('./data/train.csv')
test_df = pandas.read_csv('./data/test.csv')

class Node:
    def __init__(self, feature_index=None, feature_name=None, prediction=None, 
                is_leaf=False, children=None, target_values=None):
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.prediction = prediction
        self.is_leaf = is_leaf
        self.children = children
        self.target_values = target_values

class DecisionTree:
    def __init__ (self):
        self.root = None
        self.feature_indices = None
        self.feature_names = None

    def calculate_entropy (self, column_values: numpy.ndarray):
        _, value_counts = numpy.unique (column_values, return_counts=True)
        probabilities = value_counts / len(column_values)

        entropy = 0
        for prob in probabilities:
            entropy += -(prob * math.log(prob))

        return entropy

    def calculate_information_gain (self, train_data: pandas.core.frame.DataFrame, feature_name: str, target_feature: str):
        initial_entropy = self.calculate_entropy(train_data[target_feature].values)

        unique_values = train_data[feature_name].unique()
        weighted_entropy = 0

        for value in unique_values:
            data_subset = train_data[train_data[feature_name] == value]
            subset_entropy = self.calculate_entropy(data_subset[target_feature].values)

            weighted_entropy += (len(data_subset) / len(train_data)) * subset_entropy

        information_gain = initial_entropy - weighted_entropy
        return information_gain

    def find_best_split (self, train_data: pandas.core.frame.DataFrame, target_feature: str):
        best_information_gain = float('-inf')
        best_feature = None

        for feature in train_data.columns:
            if feature == target_feature:
                continue

            info_gain = self.calculate_information_gain(train_data, feature, target_feature)
            if (info_gain > best_information_gain):
                best_information_gain = info_gain
                best_feature = feature

        return best_information_gain, best_feature
            
    def build_tree (self, train_data: pandas.core.frame.DataFrame, target_feature: str):
        if (self.calculate_entropy(train_data[target_feature].values) == 0 \
            or train_data.shape[1] == 1):
            leaf = Node(prediction=train_data[target_feature].mode(), is_leaf=True)
            return leaf

        info_gain, split_feature = self.find_best_split(train_data, target_feature)

        children = {}
        for value in train_data[split_feature].unique():
            child_node = self.build_tree(
                                        train_data[train_data[split_feature] == value].drop(split_feature, axis=1), 
                                        target_feature
                                    )
            children[value] = child_node

        node = Node(
                    feature_index=self.feature_indices[split_feature], 
                    feature_name=split_feature, children=children, 
                    target_values=train_data[target_feature].value_counts().to_dict()
                )
        return node

    def train_model (self, train_data: pandas.core.frame.DataFrame, target_feature: str):
        self.feature_indices = {column: index for index, column in enumerate(train_data.columns)}
        self.feature_names = {index: column for index, column in enumerate(train_data.columns)}
        
        self.root = self.build_tree(train_data, target_feature)
        return self.root

    def predict_example (self, test_example):
        node = self.root
        while (not node.is_leaf):
            feature_index = node.feature_index
            node = node.children[test_example[feature_index]]

        return node.prediction

    def predict_model (self, test_data):
        predictions = []
        for index, row in test_data.iterrows():
            prediction = self.predict_example(row.values)
            predictions.append(prediction)
        return numpy.array(predictions).reshape(1, -1)

classifier = DecisionTree()
classifier.train_model(train_df, 'survived')

test_data = test_df.drop('survived', axis=1)
test_target_labels = test_df['survived'].values.reshape(1, -1)[0, :]

predictions = classifier.predict_model(test_data)[0, :]

def calculate_confusion_matrix(predictions, actual):
    true_positives = sum([pred=='yes' and label=='yes' for pred, label in zip(predictions, actual)])
    false_positives = sum([pred=='yes' and label=='no' for pred, label in zip(predictions, actual)])

    true_negatives = sum([pred=='no' and label=='no' for pred, label in zip(predictions, actual)])
    false_negatives = sum([pred=='no' and label=='yes' for pred, label in zip(predictions, actual)])

    confusion_matrix = {
        'TP' : true_positives, 
        'FP' : false_positives, 
        'TN' : true_negatives, 
        'FN' : false_negatives, 
    }

    return confusion_matrix

confusion_matrix = calculate_confusion_matrix(predictions=predictions, actual=test_target_labels)
print ("Confusion Matrix: ", confusion_matrix, "\n")

def calculate_evaluation_metrics (confusion_matrix):
    TP = confusion_matrix['TP']
    FP = confusion_matrix['FP']
    TN = confusion_matrix['TN']
    FN = confusion_matrix['FN']

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)

    return precision, recall, accuracy

precision, recall, accuracy = calculate_evaluation_metrics(confusion_matrix)

print(f'Precision: {precision:.5f}')
print(f'Recall: {recall:.5f}')
print(f'Accuracy: {accuracy:.5f}')