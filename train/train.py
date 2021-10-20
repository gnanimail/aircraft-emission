import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_model(x_train, y_train):
    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)

    # Random Forest Regression
    random_regressor = RandomForestRegressor()
    random_regressor.fit(x_train_data, y_train_data)

    joblib.dump(random_regressor, 'ae_model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()
    train_model(args.x_train, args.y_train)
