import numpy as np
import pandas as pd


def read_matrix(file_handle):
    """
    (matrix, tokenlist, category) = read_matrix(file_handle)

    Reads the file handle pointed to by 'file_handle', which is of the format of
    MATRIX.TEST, and returns a 3-tuple. The first part is 'sp_matrix',
    an m-by-n sparse matrix, where m is the number of training/testing
    examples and n is the dimension, and each row of sp_matrix consists
    of counts of word appearances. (So sp_matrix[i, j] is the number of
    times word j appears in document i.)

    tokenlist is a list of the words, where tokenlist[1] is the first
    word in the dictionary and tokenlist[end] is the last.

    category is a {0, 1}-valued vector of positive and negative
    examples. Before using in SVM code, you should transform categories
    to have signs +/-1.
    """
    it = iter(file_handle)
    headerline = next(it)
    row_col_line = next(it)
    row_col_split = row_col_line.rstrip().split()
    num_rows, num_cols = map(int, row_col_split)
    tokenlist = next(it)
    tokenlist = tokenlist.rstrip().split()

    # Now to read the matrix into the matrix. Each row represents a
    # document (mail), each column represents a distinct token. As the
    # data isn't actually that big, we just use a full matrix to save
    # time.
    full_mat = np.zeros((num_rows, num_cols), dtype=int)
    categories = np.zeros((num_rows, 1))
    for i in range(num_rows):
        str_inds = next(it)
        str_split = str_inds.rstrip().split()
        categories[i] = int(str_split[0])
        
        offset = 0
        for j in range(1, len(str_split)-1, 2):
            offset = int(str_split[j]) + offset
            count = int(str_split[j + 1])
            full_mat[i, offset] = count

    return (full_mat, tokenlist, categories)


def read_data(filename):
    with open(filename) as handle:
        matrix, tokenlist, categories = read_matrix(handle)
        matrix = np.concatenate((categories, matrix), axis=1)
        tokenlist = ['SPAM'] + tokenlist
        df = pd.DataFrame(matrix, columns=tokenlist, dtype=int)

        return df
