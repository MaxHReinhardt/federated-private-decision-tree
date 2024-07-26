from src.federated_private_ID3 import *
from src.ID3 import *
from src.explore_tree import *
from src.preprocessing import *

import itertools
import numpy as np
from sklearn.model_selection import train_test_split
import time


def compare_settings(setting_type_list, num_clients_list, seed_list, data_path, min_samples_split):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    df = preprocessing(df)

    # Get basic information about data
    attributes = sorted(df.columns[:-1].tolist())
    values_dict = {col: df[col].unique().tolist() for col in df.columns}
    classes = df['Segmentation'].unique().tolist()

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # General algorithm settings
    min_samples_split = min_samples_split

    # Initialize result list
    results_list = []
    # results = pd.DataFrame(columns=['setting_type', 'num_clients', 'seed', 'f1', 'accuracy', 'execution_time'])

    for setting_type in setting_type_list:
        if setting_type == 'centralized':
            start_time = time.time()
            tree = id3(train_df, attributes, values_dict, min_samples_split=min_samples_split)
            end_time = time.time()
            execution_time = end_time - start_time  # Calculate the execution time
            f1, accuracy, _ = evaluate_on_testset(tree, test_df)
            results_row = {
                'setting_type': setting_type,
                'num_clients': -99,
                'seed': -99,
                'f1': f1,
                'accuracy': accuracy,
                'execution_time': execution_time
            }
            results_list.append(results_row)
            print(f'Completed -- setting_type: {setting_type}.')

        else:
            if setting_type == 'federated_standard':
                private = False
            elif setting_type == 'federated_private':
                private = True
            else:
                private = False
                print('The setting_type is not valid, proceed with federated_standard.')

            for num_clients, seed in itertools.product(num_clients_list, seed_list):
                client_datasets = split_dataset_in_n_equally_sized_client_datasets(train_df, num_clients, seed=seed)
                start_time = time.time()
                tree = federated_private_id3(clients_data=client_datasets,
                                             attributes=attributes,
                                             values_dict=values_dict,
                                             classes=classes,
                                             default_class=None,
                                             min_samples_split=min_samples_split,
                                             private=private)
                end_time = time.time()
                execution_time = end_time - start_time  # Calculate the execution time
                f1, accuracy, _ = evaluate_on_testset(tree, test_df)
                results_row = {
                    'setting_type': setting_type,
                    'num_clients': num_clients,
                    'seed': seed,
                    'f1': f1,
                    'accuracy': accuracy,
                    'execution_time': execution_time
                }
                results_list.append(results_row)
                print(f'Completed -- setting_type: {setting_type}, num_clients: {num_clients}, seed: {seed}.')

    results = pd.DataFrame(results_list)
    return results
