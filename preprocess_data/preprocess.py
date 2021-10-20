from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def _preprocess_data():
     subset_ = pd.read_csv('../data/subsetted.csv', encoding='cp1252')
     X = subset_.drop('Nox_EI_C/O', axis=1)
     y = subset_.iloc[:, -1]

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

     np.save('x_train.npy', X_train)
     np.save('x_test.npy', X_test)
     np.save('y_train.npy', y_train)
     np.save('y_test.npy', y_test)
     
if __name__ == '__main__':
     print('Preprocessing data...')
     _preprocess_data()
