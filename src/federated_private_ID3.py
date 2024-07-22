import math
import pandas as pd
from collections import defaultdict

# TODO for extending to private version:
## implement private summation protocol and use in the right parts
## Re-check for possible information leaks


def privacy_preserving_secret_summation():
    pass


# Client label counts
def client_count_labels(data, classes):
    client_label_counts = defaultdict(int)
    for label in classes:
        client_label_counts[label] = data[data.iloc[:, -1] == label].shape[0]
    return client_label_counts


### Extend to private version here ###
# Calculate joint label counts of all clients
def aggregate_label_counts(client_label_counts):
    total_label_counts = defaultdict(int)
    for label_counts in client_label_counts:
        for label, count in label_counts.items():
            total_label_counts[label] += count
    return total_label_counts


# Calculate entropy of a dataset
def entropy(counts):
    total_count = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            probability = count / total_count
            entropy -= probability * math.log2(probability)
    return entropy


# Calculate the information gain
def information_gain(total_entropy, subsets_counts):
    total_count = sum(sum(counts.values()) for counts in subsets_counts)
    weighted_entropy = 0.0

    for counts in subsets_counts:
        subset_entropy = entropy(counts)
        subset_count = sum(counts.values())
        weighted_entropy += (subset_count / total_count) * subset_entropy

    gain = total_entropy - weighted_entropy
    return gain


# Each client calculates the counts for potential splits
def client_calculate_split_counts(data, attributes, values_dict, classes):
    split_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for attribute in attributes:
        for value in values_dict[attribute]:
            subset = data.loc[data[attribute] == value]
            for label in classes:
                split_counts[attribute][value][label] += subset[subset.iloc[:, -1] == label].shape[0]

    return split_counts


### Extend to private version here ###
def aggregate_split_counts(client_split_counts):
    total_split_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Aggregate counts from all clients
    for client_split in client_split_counts:
        for attr, value_counts in client_split.items():
            for value, label_counts in value_counts.items():
                for label, count in label_counts.items():
                    total_split_counts[attr][value][label] += count

    return total_split_counts


# Central instance determines the best split
def choose_split(total_split_counts, total_label_counts):
    best_gain = -float('inf')
    best_attr = None

    total_entropy = entropy(total_label_counts)

    for attr, value_counts in total_split_counts.items():
        subsets_counts = [label_counts for label_counts in value_counts.values()]
        gain = information_gain(total_entropy, subsets_counts)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    return best_attr


# Each client partitions their data according to the best split
def client_partition_data(data, split_attribute, values_dict):
    partitions = {}
    for value in values_dict[split_attribute]:
        subset = data.loc[data[split_attribute] == value]
        partitions[value] = subset
    return partitions


# Fit the decision tree in a federated setting
def federated_private_id3(clients_data, attributes, values_dict, classes, default_class=None, min_samples_split=2):
    # Clients calculate their individual label counts
    client_label_counts = [client_count_labels(data, classes) for data in clients_data]
    # Sum up client label counts to obtain total label counts
    total_label_counts = aggregate_label_counts(client_label_counts)
    # Sorting to achieve order invariance
    sorted_total_label_counts = {key: total_label_counts[key] for key in sorted(total_label_counts.keys())}

    # If the dataset is empty or attributes are exhausted, return the default class
    if sum(sorted_total_label_counts.values()) == 0 or not attributes:
        return default_class
    # If all labels are the same, return that label
    if len(sorted_total_label_counts) == 1:
        return next(iter(sorted_total_label_counts))
    # If the number of samples is less than the minimum, return the majority class
    if sum(sorted_total_label_counts.values()) < min_samples_split:
        return max(sorted_total_label_counts, key=sorted_total_label_counts.get)
    # If there are no features left, return the majority class (in the next recursion)
    default_class = max(sorted_total_label_counts, key=sorted_total_label_counts.get)

    # Clients calculate split counts
    client_split_counts = [client_calculate_split_counts(data, attributes, values_dict, classes) for data in clients_data]
    # Sum up client split counts to obtain total split counts
    total_split_counts = aggregate_split_counts(client_split_counts)
    # Sorting to achieve order invariance
    sorted_total_split_counts = {key: total_split_counts[key] for key in sorted(total_split_counts.keys())}
    # Choose split based on aggregated counts
    best_attr = choose_split(sorted_total_split_counts, total_label_counts)

    # Clients partition their data according to the best split
    client_partitions = [client_partition_data(data, best_attr, values_dict) for data in clients_data]

    # Add best attribute as next split to the tree
    tree = {best_attr: {}}
    # Reduce list of remaining attributes
    remaining_attributes = [attr for attr in attributes if attr != best_attr]

    # Recurse for each partition
    for value in values_dict[best_attr]:
        partitions = [partition[value] for partition in client_partitions if value in partition]
        subtree = federated_private_id3(partitions, remaining_attributes, values_dict, classes, default_class, min_samples_split)
        tree[best_attr][value] = subtree

    return tree
