import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from methods.goa_svm import GOA_SVM


def main():
    data = pd.read_csv('data/gpt.csv')

    x = data.iloc[:,:273]
    y = data['technique']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    verbose = True
    lb = [1, 1e-02]
    ub = [500, 5e-02]
    pop_size = 30
    epoch = 10
    c_minmax = (0.00001, 5)
    md = GOA_SVM(lb=lb, ub=ub, verbose=verbose, pop_size=pop_size, epoch=epoch, c_minmax=c_minmax)
    best_pos, best_fit, _ = md.train(X_train, y_train)
    print(best_pos)
    print(best_fit)


if __name__ == '__main__':
    main()