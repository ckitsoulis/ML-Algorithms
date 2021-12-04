import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# Create a bootstrapped dataset
def bootstraping(dataset):
    
    n_samples = len(dataset)

    random_indices = np.random.randint(low = 0, high = n_samples, size = n_samples)

    return dataset.iloc[random_indices,:]

# Bulid Decision Tree using sklearn with built-in arguments: min_samples_leaf and max_features = sqrt(n_features)
def Build_DecisionTree(X, Y, min_samples_leaf = 1):

    tree = DecisionTreeClassifier(min_samples_leaf = min_samples_leaf, max_features="sqrt").fit(X,Y)

    return tree

# Training of Random Forest
def TrainRF(X, Y, n_trees, min_sample_leaf = 1):

    X_Y = pd.concat([X,Y], axis=1) # concatenate X-Y for bootstrapping procedure

    forest = []

    for n in range(n_trees):

        bootstrapped_dataset = bootstraping(X_Y)

        # Train each Decision Tree
        tree = Build_DecisionTree(bootstrapped_dataset.iloc[:,:-1], bootstrapped_dataset.iloc[:,-1], min_sample_leaf)

        forest.append(tree)

    return forest

# Predict the class of samples based on the trained model
def PredictRF(model, dataset, each_tree_prediction = None):
    
    predictions = dict()
    for number_, model_ in enumerate(model):
        # iterate on every tree & make predictions for dataset
        name = "tree_{}".format(number_)
        tree_prediction = model_.predict(dataset)

        # save predictions of each tree in dictionary
        predictions[name] = tree_prediction
    
    predictions_dataframe = pd.DataFrame(predictions)

    if each_tree_prediction:

        return predictions_dataframe.mode(axis = 1)[0], predictions_dataframe  #keep the most common prediction for each sample (maximum voting), return predictions by any tree
    
    else:

        return predictions_dataframe.mode(axis = 1)[0] #keep the most common prediction for each sample (maximum voting)

# estimate accuracy of predictions
def accuracy (predictions, test_Y):

    return np.sum(predictions.to_numpy() == test_Y.to_numpy())/ len(test_Y)

# Split the dataset to Train and Test for a given ratio (e.g. 0.7 --> 70%)
def split_dataset(dataset, ratio):
    n_rows = len(dataset)

    split = int(ratio*n_rows)

    train_df, test_df = dataset.iloc[:split,:], dataset.iloc[split:,:]

    return train_df, test_df
