import numpy as np
import pandas as pd


def read_csv(csv_handle):
    raw_data = np.loadtxt(csv_handle, delimiter=',')

    try:
        cols = raw_data.shape[1]
    except:
        raise ValueError('Wrong data shape. Got shape {}.'\
                         'Need a shape of the form (Rows,Cols).'
                         .format(raw_data.shape))

    col_names = ['y'] 
    for i in range(cols-1):
        col_names.append('x{}'.format(i))

    return pd.DataFrame(raw_data, columns=col_names)


def load_dataset(csv_handle_tr, csv_handle_te):
    training_data = read_csv(csv_handle_tr)
    testing_data = read_csv(csv_handle_te)

    # In order for the testing and training data to appear valid, it should appear
    # to come from the data data set. So it necessarily must have the same number of
    # columns and the same column labels.
    if (training_data.shape[1] != testing_data.shape[1]):
        raise ValueError(
            ('The training and testing data do not have the same number of\n'
             'columns. These do not appear to be from the same data set.\n'
             'Training data shape: {}\n'
             'Testing data shape: {}\n'
            ).format(training_data.shape, testing_data.shape))

    if not training_data.columns.equals(testing_data.columns):
        raise ValueError(
            ('Training and testing data have mismatched columns.'
             'Training data columns: {}\n'
             'Testing data columns: {}\n'
            ).format(training_data.columns, testing_data.columns))

    return training_data, testing_data

