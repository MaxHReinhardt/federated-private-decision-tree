import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import graphviz

#
# Predict
#

# Make a prediction for an individual instance
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    if attribute in instance:
        value = instance[attribute]
        if value in tree[attribute]:
            return predict(tree[attribute][value], instance)
    return None  # Return None if no prediction is possible


# Make a prediction for all instances in a pandas df
def predict_all(tree, df):
    return df.apply(lambda x: predict(tree, x), axis=1).tolist()


#
# Evaluate
#

# Calculate micro-f1 and accuracy based on testset
def evaluate_on_testset(tree, test_df):
    true_labels = test_df.iloc[:, -1]
    predictions = predict_all(tree, test_df)
    f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)
    confusion_mtrx = confusion_matrix(true_labels, predictions)

    return f1, accuracy, confusion_mtrx


#
# Visualize
#

# Visualize a decision tree
def visualize_tree(tree, tree_name="decision_tree", format="png", cleanup=True):
    def add_nodes_edges(tree, dot=None, parent_name=None, edge_label=""):
        if dot is None:
            dot = graphviz.Digraph(tree_name)

        if isinstance(tree, dict):
            for attribute, branches in tree.items():
                node_name = f"node{add_nodes_edges.counter}"
                add_nodes_edges.counter += 1
                dot.node(name=node_name, label=str(attribute))

                if parent_name is not None:
                    dot.edge(parent_name, node_name, label=edge_label)

                for value, subtree in branches.items():
                    add_nodes_edges(subtree, dot=dot, parent_name=node_name, edge_label=str(value))
        else:
            leaf_name = f"leaf{add_nodes_edges.counter}"
            add_nodes_edges.counter += 1
            dot.node(name=leaf_name, label=str(tree), shape='box')
            if parent_name is not None:
                dot.edge(parent_name, leaf_name, label=edge_label)

        return dot

    add_nodes_edges.counter = 0
    dot = add_nodes_edges(tree)
    dot.render(tree_name, format=format, cleanup=cleanup)
    return dot

