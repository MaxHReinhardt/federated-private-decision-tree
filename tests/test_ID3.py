from src.ID3 import *
from src.explore_tree import *
from src.preprocessing import *


def test_ID3_workflow():

    # Load and preprocess data
    df = pd.read_csv('../data/Small.csv')
    df = preprocessing(df)
    print(df.head())

    # Fit the decision tree
    attributes = sorted(df.columns[:-1].tolist())
    values_dict = {col: df[col].unique().tolist() for col in df.columns}
    min_samples_split = 5
    tree = id3(df, attributes, values_dict, min_samples_split=min_samples_split)
    print(f'Decision tree dict: {tree}')

    # Calculate evaluation metrics (on the same df used for training for simplicity)
    f1, accuracy, confusion_mtrx = evaluate_on_testset(tree, df)
    print(f'F1 score: {f1}. Accuracy: {accuracy}.')
    print(f'Confusion Matrix: {confusion_mtrx}')

    # Visualize the tree
    _ = visualize_tree(tree, tree_name="graphics/decision_tree", format="png", cleanup=True)


