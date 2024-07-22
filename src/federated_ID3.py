import math
import pandas as pd
from collections import defaultdict


# TODO: Check extendability to private version. The algorithms should align but only vary by the addition of secure multi-party computation.

# TODO for extending to private version:
## Split central_aggregate_and_choose_split into two functions
## See in-code to dos
## Handle empty client datasets differently: They have to report 0 in private summation, not disappear. Otherwise, the information would be leaked that no such instance is present.
## Re-check for possible information leaks
## implement private summation protocol


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
def client_calculate_splits(data, attributes):
    split_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for attr in attributes:
        for value in data[attr].unique():
            subset = data[data[attr] == value]
            label_counts = subset.iloc[:, -1].value_counts()
            for label, count in label_counts.items():
                split_counts[attr][value][label] += count

    return split_counts


# Central instance aggregates counts and determines the best split
def central_aggregate_and_choose_split(client_splits, total_label_counts):
    aggregate_splits = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    ### Extend to private version here ###
    # Aggregate counts from all clients
    for client_split in client_splits:
        for attr, value_counts in client_split.items():
            for value, label_counts in value_counts.items():
                for label, count in label_counts.items():
                    aggregate_splits[attr][value][label] += count

    best_gain = -float('inf')
    best_attr = None

    total_entropy = entropy(total_label_counts)

    # Sorting to achieve order invariance
    sorted_aggregate_splits = {key: aggregate_splits[key] for key in sorted(aggregate_splits.keys())}

    for attr, value_counts in sorted_aggregate_splits.items():
        subsets_counts = [label_counts for label_counts in value_counts.values()]
        gain = information_gain(total_entropy, subsets_counts)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    return best_attr


# Each client partitions their data according to the best split
def client_partition_data(data, split_attribute):
    partitions = {}
    for value in data[split_attribute].unique():
        subset = data[data[split_attribute] == value]
        if subset.empty:
            continue
        partitions[value] = subset
    return partitions


# Fit the decision tree in a federated setting
def federated_id3(clients_data, attributes, default_class=None, min_samples_split=2):
    ### Checking for empty client data is not privacy preserving. Solution: calculate total_label_counts and check for all 0 instead. ###
    # If the dataset is empty or attributes are exhausted, return the default class
    if not clients_data or not attributes:
        return default_class

    ### Extend to private version here ###
    # Calculate total label counts across all clients
    total_label_counts = defaultdict(int)
    for data in clients_data:
        label_counts = data.iloc[:, -1].value_counts()
        for label, count in label_counts.items():
            total_label_counts[label] += count

    # Sorting to achieve order invariance
    sorted_total_label_counts = {key: total_label_counts[key] for key in sorted(total_label_counts.keys())}

    # If all labels are the same, return that label
    if len(sorted_total_label_counts) == 1:
        return next(iter(sorted_total_label_counts))
    # If the number of samples is less than the minimum, return the majority class
    if sum(sorted_total_label_counts.values()) < min_samples_split:
        return max(sorted_total_label_counts, key=sorted_total_label_counts.get)
    # If there are no features left, return the majority class (in the next recursion)
    default_class = max(sorted_total_label_counts, key=sorted_total_label_counts.get)

    # Clients calculate split counts
    client_splits = [client_calculate_splits(data, attributes) for data in clients_data]

    # Central instance aggregates counts and chooses the best split
    best_attr = central_aggregate_and_choose_split(client_splits, sorted_total_label_counts)

    tree = {best_attr: {}}

    # Clients partition their data according to the best split
    client_partitions = [client_partition_data(data, best_attr) for data in clients_data]

    remaining_attributes = [attr for attr in attributes if attr != best_attr]

    # Recurse for each partition
    ### Unique values must be determined privately, however, there is no need to determine the unique values in each recursion. ###
    ### Might be passed as an argument, or determined within central_aggregate_and_choose_split ###
    unique_values = set(val for data in clients_data for val in data[best_attr].unique())
    for value in unique_values:
        partitions = [partition[value] for partition in client_partitions if value in partition]
        subtree = federated_id3(partitions, remaining_attributes, default_class, min_samples_split)
        tree[best_attr][value] = subtree

    return tree
