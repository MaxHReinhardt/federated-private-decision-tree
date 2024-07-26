import math
import random
from sympy import symbols, Eq, solve
import pandas as pd
from collections import defaultdict


def privacy_preserving_secret_summation(secrets):
    # 0 Initialization
    num_parties = len(secrets)
    random_values = random.sample(range(1, num_parties*10), num_parties) # each client gets assigned a public random value
    degree_of_polynomials = num_parties - 1
    # For each client, generate a random polynomial
    polynomials = [generate_random_polynomial(degree_of_polynomials, secret) for secret in secrets]

    # 1 Distribution phase: Compute shares and send to corresponding clients
    distribution_list = []
    # loop over clients j
    for value in random_values:
        # each client i sends a share to client j, produced using polynomial_{client_i} and public value_{client_j}
        received_shares = [evaluate_polynomial(coefficients, value) for coefficients in polynomials]
        distribution_list.append(received_shares)

    # 2 Intermediate computation: Each client sums up all received shares and shares the result with all other clients
    interres = [sum(received_shares) for received_shares in distribution_list]

    # 3 Final computation phase: Solve the system of equations to obtain the SUM of secrets
    # Each interres/random-value combination forms an equation of the following form:
    # interres_{i} = (sum of coefficients)*x_{i}^{n-1} + ... + sum_of_secrets
    equations = []
    for i in range(num_parties):
        eq = Eq(sum(symbols(f'a{k}') * (random_values[i] ** k) for k in range(degree_of_polynomials+1)), interres[i])
        equations.append(eq)

    solution = solve(equations, dict=True)
    sum_of_secrets = list(solution[0].values())[0]

    return sum_of_secrets


def generate_random_polynomial(degree, secret):
    coefficients = [secret] + [random.randint(-10, 10) for _ in range(degree-1)]
    return coefficients


def evaluate_polynomial(coefficients, x):
    # Assuming the secret is the first coefficient
    return sum(c * (x ** i) for i, c in enumerate(coefficients))


def client_count_labels(data, classes):
    client_label_counts = defaultdict(int)
    for label in classes:
        client_label_counts[label] = data[data.iloc[:, -1] == label].shape[0]
    return client_label_counts


def aggregate_label_counts(client_label_counts, private):
    # Collect label counts of all clients for summation
    collected_label_counts = defaultdict(list)
    for label_counts in client_label_counts:
        for label, count in label_counts.items():
            collected_label_counts[label].append(count)

    # Sum counts privately or with standard sum operation
    aggregated_label_counts = defaultdict(int)
    if private:
        for label, counts_list in collected_label_counts.items():
            aggregated_label_counts[label] = privacy_preserving_secret_summation(counts_list)
    else:
        for label, counts_list in collected_label_counts.items():
            aggregated_label_counts[label] = sum(counts_list)

    return aggregated_label_counts


# Calculate the label counts for all possible splits (individual client perspective)
def client_calculate_split_counts(data, attributes, values_dict, classes):
    split_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for attribute in attributes:
        for value in values_dict[attribute]:
            subset = data.loc[data[attribute] == value]
            for label in classes:
                split_counts[attribute][value][label] += subset[subset.iloc[:, -1] == label].shape[0]

    return split_counts


def aggregate_split_counts(client_split_counts, private):
    # Collect split counts of all clients for summation
    collected_split_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for client_split in client_split_counts:
        for attr, value_counts in client_split.items():
            for value, label_counts in value_counts.items():
                for label, count in label_counts.items():
                    collected_split_counts[attr][value][label].append(count)

    # Sum counts privately or with standard sum operation
    aggregated_split_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    if private:
        for attr, value_counts in collected_split_counts.items():
            for value, label_counts in value_counts.items():
                for label, counts_list in label_counts.items():
                    aggregated_split_counts[attr][value][label] = privacy_preserving_secret_summation(counts_list)
    else:
        for attr, value_counts in collected_split_counts.items():
            for value, label_counts in value_counts.items():
                for label, counts_list in label_counts.items():
                    aggregated_split_counts[attr][value][label] = sum(counts_list)

    return aggregated_split_counts


def entropy(counts):
    total_count = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            probability = count / total_count
            entropy -= probability * math.log2(probability)
    return entropy


def information_gain(total_entropy, subsets_counts):
    total_count = sum(sum(counts.values()) for counts in subsets_counts)
    weighted_entropy = 0.0

    for counts in subsets_counts:
        subset_entropy = entropy(counts)
        subset_count = sum(counts.values())
        weighted_entropy += (subset_count / total_count) * subset_entropy

    gain = total_entropy - weighted_entropy
    return gain


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


def client_partition_data(data, split_attribute, values_dict):
    partitions = {}
    for value in values_dict[split_attribute]:
        subset = data.loc[data[split_attribute] == value]
        partitions[value] = subset
    return partitions


# Federated private ID3
def federated_private_id3(clients_data, attributes, values_dict, classes,
                          default_class=None, min_samples_split=2, private=False):
    # Clients calculate their individual label counts
    client_label_counts = [client_count_labels(data, classes) for data in clients_data]
    # Sum up client label counts to obtain total label counts
    total_label_counts = aggregate_label_counts(client_label_counts, private=private)
    # Sorting to achieve order invariance
    sorted_total_label_counts = {key: total_label_counts[key] for key in sorted(total_label_counts.keys())}

    # If the dataset is empty or attributes are exhausted, return the default class
    if sum(sorted_total_label_counts.values()) == 0 or not attributes:
        return default_class
    # If all labels are the same, return that label
    if sum(1 for value in sorted_total_label_counts.values() if value > 0) == 1:
        return next(label for label, value in sorted_total_label_counts.items() if value > 0)
    # If the number of samples is less than the minimum, return the majority class
    if sum(sorted_total_label_counts.values()) < min_samples_split:
        return max(sorted_total_label_counts, key=sorted_total_label_counts.get)
    # If there are no features left, return the majority class (in the next recursion)
    default_class = max(sorted_total_label_counts, key=sorted_total_label_counts.get)

    # Clients calculate split counts
    client_split_counts = [client_calculate_split_counts(data, attributes, values_dict, classes) for data in clients_data]
    # Sum up client split counts to obtain total split counts
    total_split_counts = aggregate_split_counts(client_split_counts, private=private)
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
        subtree = federated_private_id3(partitions, remaining_attributes, values_dict, classes,
                                        default_class, min_samples_split, private)
        tree[best_attr][value] = subtree

    return tree
