# imports
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    # Mean normalization for X
    X_mean = np.mean(X, axis=0)
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    X_normalized = (X - X_mean) / (X_max - X_min)

    # Mean normalization for y
    y_mean = np.mean(y, axis=0)
    y_max = np.max(y, axis=0)
    y_min = np.min(y, axis=0)
    y_normalized = (y - y_mean) / (y_max - y_min)
    return X_normalized, y_normalized

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    # Ensure X is two-dimensional
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    # Create a column of ones with the same number of rows as X
    ones = np.ones((X.shape[0], 1))
    # Horizontally stack the column of ones to the left of the original feature matrix X
    X = np.hstack((ones, X))
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.

    m = len(y)
    predictions = X.dot(theta)
    errors = (predictions -y) ** 2
    J = np.sum(errors) / (2*m)

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration


    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = X.T.dot(errors) / len(y)
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    XtX = X.T.dot(X)
    XtX_inv = np.linalg.inv(XtX)
    pinv_X = XtX_inv.dot(X.T)
    pinv_theta = pinv_X.dot(y)
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = X.T.dot(errors) / len(y)
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X, y, theta))

        if len(J_history) >= 2 and J_history[-2] - J_history[-1] < 1e-8:
            break
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}

    #choosing random theta
    np.random.seed(42)  # Setting the seed
    n_features = X_train.shape[1]
    theta = np.random.rand(n_features)

    #finding dictionary
    for alpha in alphas:
        model, j_history = efficient_gradient_descent(X_train, y_train, theta,alpha, iterations)
        cost = compute_cost(X_val, y_val, model)
        alpha_dict[alpha] = cost
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []

    all_features = set(range(X_train.shape[1]))
    while len(selected_features) < 5:
        min_cost = np.inf
        best_feature = None

        for feature in all_features - set(selected_features):
            feature_subset = selected_features + [feature]

            # Extract the relevant columns from X_train and X_val
            X_train_subset = X_train[:, feature_subset]
            X_val_subset = X_val[:, feature_subset]

            # Apply the bias trick to the subsets
            X_train_subset = apply_bias_trick(X_train_subset)
            X_val_subset = apply_bias_trick(X_val_subset)

            # choosing random theta
            np.random.seed(42)  # Setting the seed
            n_features = X_train_subset.shape[1]
            theta = np.random.rand(n_features)

            theta, ignore = efficient_gradient_descent(X_train_subset, y_train, theta, best_alpha, iterations)
            cost = compute_cost(X_val_subset, y_val, theta)
            if(cost < min_cost):
                min_cost = cost
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()

    new_features = {}  # Use a dictionary to store new features
    features = list(df.columns)

    for feature in features:
        new_features[f'{feature}^2'] = df[feature] ** 2

    n = len(df.columns)
    feature = list(df.columns)
    for i in range(n):
        for j in range(i+1, n):
            feature1, feature2 = features[i], features[j]
            new_features[f'{feature1}*{feature2}'] = df[feature1] * df[feature2]

    new_features_df = pd.DataFrame(new_features)
    df_poly = pd.concat([df_poly, new_features_df], axis=1)

    return df_poly