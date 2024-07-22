from src.federated_ID3 import *
from src.explore_tree import *
from src.preprocessing import *


def test_federated_ID3_workflow():

    # np.random.seed(0)

    # Load and preprocess data
    df = pd.read_csv('../data/Small.csv')
    df = preprocessing(df)
    client_datasets = split_dataset_in_n_equally_sized_client_datasets(df, 10)
    # client_datasets = [df]
    print(client_datasets[0].head())

    # Fit the decision tree
    attributes = sorted(client_datasets[0].columns[:-1].tolist())
    min_samples_split = 5
    tree = federated_id3(client_datasets, attributes, default_class=None, min_samples_split=min_samples_split)
    print(f'Decision tree dict: {tree}')

    # Calculate evaluation metrics (on the same df used for training for simplicity)
    f1, accuracy, confusion_mtrx = evaluate_on_testset(tree, df)
    print(f'F1 score: {f1}. Accuracy: {accuracy}.')
    print(f'Confusion Matrix: {confusion_mtrx}')

    # Visualize the tree
    _ = visualize_tree(tree, tree_name="graphics/decision_tree", format="png", cleanup=True)

