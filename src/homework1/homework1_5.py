import numpy as np
import numpy.linalg as linalg
import pandas as pd


def linear_regression(X, y):
    return linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def go():
    data = np.loadtxt('quasar_train.csv', delimiter=',')
    wavelengths = data[0]
    fluxes = data[1]
    ones = np.ones(fluxes.size)

    df_ones = pd.DataFrame(ones, columns=['xint'])
    df_wavelengths = pd.DataFrame(wavelengths, columns=['wavelength'])
    df_fluxes = pd.DataFrame(fluxes, columns=['flux'])

    df = pd.concat([df_ones, df_wavelengths, df_fluxes], axis=1)
    X = pd.concat([df['xint'], df['wavelength']], axis=1)
    y = df['flux']

    return linear_regression(X, y)


def main():
    theta = go()

    print('theta = {}'.format(theta))