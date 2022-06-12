import numpy as np

def get_batches(x, y, batch_size=None):
    "Utility function to create batches of a dataset"
    n_data = x.shape[0]
    # shuffle data
    indices = np.arange(n_data)
    np.random.shuffle(indices)
   
    if batch_size is None:
        batch_size = 1
    # create a generator of batches
    for b_idx in range(int(np.ceil(n_data/batch_size))):
        yield x[b_idx*batch_size:min((b_idx+1)*batch_size, n_data)], \
             y[b_idx*batch_size:min((b_idx+1)*batch_size, n_data)]

def get_accuracy(output, target):
    "Returns accuracy of prediction"
    return (output == target).mean()

def mse(output, target):
    "Returns accuracy of prediction"
    return (np.square(output-target)).mean()

# Standardize and Splitting Dataset
def standardize(data_x):
    """
    Standardize input data (only independent variables)
    
    Args:
        data_x (pandas.DataFrame)
    Return:
        std_data_x : standardized data 
    """
    std_data_x = data_x.copy()
    for column in std_data_x.columns.to_list():
        std_data_x[column] = (std_data_x[column] - std_data_x[column].mean()) / std_data_x[column].std()

    return std_data_x

def train_test_split(data_x, data_y, test_ratio=0.2):
    """
    Splits input dataset into training and testing sets.

    Args:
        data_x : independent variables of data
        data_y : dependent variable
        test_ratio : split ratio for test data

    Return:
        (train_data_x, train_data_y) : training data
        (test_data_x, test_data_y) : testing data
    """
    n_data = data_x.shape[0]
    print("Total Number of datapoints: {}".format(n_data))
    n_train_rows = np.ceil(n_data*(1.-test_ratio)).astype(int)

    shuffled_indices = np.random.permutation(n_data)
    train_indices = shuffled_indices[:n_train_rows]
    test_indices = shuffled_indices[n_train_rows:]

    train_data_x, train_data_y = data_x.iloc[train_indices], data_y.iloc[train_indices]
    test_data_x, test_data_y = data_x.iloc[test_indices], data_y.iloc[test_indices]

    print("Num. of training datapoints: {} | Num. of testing datapoints: {}".format(train_data_x.shape[0], test_data_x.shape[0]))
    return (train_data_x, train_data_y), (test_data_x, test_data_y)    