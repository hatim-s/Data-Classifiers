import numpy
import pandas
import math

train_df = pandas.read_csv('./data/train.csv')
test_df = pandas.read_csv('./data/test.csv')

class NaiveBayesClassifier:
	def __init__(self, alpha=1):
		self.feature_probs = None
		self.target_probs = None
		self.alpha = alpha

	def calculate_prior_probabilities (self, target_values: numpy.ndarray):
		unique_targets, target_counts = numpy.unique(target_values, return_counts=True)

		total_targets = len(target_values)
		target_probs = {target : (count + 1) / (total_targets + 2)  
						for target, count in zip(unique_targets, target_counts)}

		return target_probs

	def calculate_feature_probabilities (self, train_data: pandas.core.frame.DataFrame, 
										target_column: str):
		feature_probabilities = {}
		for target_class in train_data[target_column].unique():
			mask = train_data[target_column] == target_class
			masked_df = train_data[mask].drop(target_column, axis=1)

			for column in masked_df:
				unique_values = masked_df[column].unique()
				value_counts = masked_df[column].value_counts()

				for value in unique_values:
					feature_probabilities[(value, target_class)] = (
						(value_counts[value] + self.alpha) / 
						(len(masked_df) + self.alpha * len(unique_values))
					)
					
		return feature_probabilities

	def train_model(self, train_data: pandas.core.frame.DataFrame, target_feature: str):
		self.target_probs = self.calculate_prior_probabilities (target_values=train_data.survived)
		self.feature_probs = self.calculate_feature_probabilities (train_data=train_data, target_column='survived')

		with open('./out/probabilities.txt', 'w') as file:
			for key, value in self.target_probs.items():
				file.write(f"{key} : {value}\n")

		with open('./out/probabilities.txt', 'a') as file:
			for key, value in self.feature_probs.items():
				file.write(f"{key[0]}, {key[1]} : {value}\n")

	def predict_target_value(self, test_features: numpy.ndarray):
		max_prob = float('-inf')
		target_class = None
		for target_val, p_target in self.target_probs.items():
			# denotes the probability of the feature, given class
			p_feature_class = 1 
			for feat in test_features:
				p_feature_class *= self.feature_probs.get((feat, target_val), 0)

			prob = (p_target * p_feature_class)

			if (prob > max_prob):
				max_prob = prob
				target_class = target_val

		return target_class

	def predict_model(self, test_data: pandas.core.frame.DataFrame):
		predictions = [self.predict_target_value(row.values) \
							for _, row in test_data.iterrows()]
		
		return predictions

# Creating and Training Classifier

classifier = NaiveBayesClassifier()
classifier.train_model(train_df, 'survived')

test_data = test_df.drop("survived", axis=1)
test_target_labels = test_df['survived'].values.reshape(1, -1)[0, :]

test_target_pred = classifier.predict_model(test_data)

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

confusion_matrix = calculate_confusion_matrix(predictions=test_target_pred, 
												actual=test_target_labels)
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