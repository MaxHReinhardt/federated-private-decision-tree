import math
import pandas as pd


# Calculate entropy of a dataset
def entropy(data):
    label_counts = data.iloc[:, -1].value_counts()
    total_count = len(data)
    entropy = 0.0
    for count in label_counts:
        probability = count / total_count
        entropy -= probability * math.log2(probability)
    return entropy


# Calculate the information gain
def information_gain(data, attribute):
    total_entropy = entropy(data)
    attribute_values = data[attribute].unique()
    weighted_entropy = 0.0
    total_count = len(data)

    for value in attribute_values:
        subset = data[data[attribute] == value]
        subset_entropy = entropy(subset)
        weighted_entropy += (len(subset) / total_count) * subset_entropy

    gain = total_entropy - weighted_entropy
    return gain


# Choose the best attribute for the next split
def best_attribute(data, attributes):
    best_gain = -float('inf')
    best_attr = None

    # Sorting to achieve order invariance
    sorted_attributes = sorted(attributes)

    for attr in sorted_attributes:
        gain = information_gain(data, attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr


# Fit the decision tree
def id3(data, attributes, values_dict, default_class=None, min_samples_split=2):
    labels = data.iloc[:, -1]
    # If the dataset is empty or attributes are exhausted, return the default class
    if data.empty or not attributes:
        return default_class
    # If all labels are the same, return that label
    if labels.nunique() == 1:
        return labels.iloc[0]
    # If the number of samples is less than the minimum, return the majority class
    if len(data) < min_samples_split:
        return labels.mode()[0]
    # If there are no features left, return the majority class (in the next recursion)
    default_class = labels.mode()[0]

    best_attr = best_attribute(data, attributes)
    tree = {best_attr: {}}

    remaining_attributes = [attr for attr in attributes if attr != best_attr]

    for value in values_dict[best_attr]:
        subset = data[data[best_attr] == value]
        subtree = id3(subset, remaining_attributes, values_dict, default_class, min_samples_split)
        tree[best_attr][value] = subtree

    return tree

