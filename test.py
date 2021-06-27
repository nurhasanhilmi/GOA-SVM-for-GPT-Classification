import numpy as np
import pandas as pd
import time

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from methods.goa_svm import GOA_SVM
from methods.svm import SVM

def main_goa():
    data = pd.read_csv('data/gpt_split.csv')

    X_train = data.iloc[:,:273]
    y_train = data['technique']

    verbose = True
    lb = [1, 0.01]
    ub = [10, 0.1]
    pop_size = 3
    epoch = 1
    k_fold = 2
    md = GOA_SVM(k_fold=k_fold, lb=lb, ub=ub, verbose=verbose, pop_size=pop_size, epoch=epoch)
    best_pos, best_fit, _ = md.train(X_train, y_train)
    print('GLOBAL BEST POSITION : ', best_pos)
    print('GLOBAL BEST FITNESS : ', best_fit)

def main_svm():
    # data = pd.read_csv('data/gpt_split_sample.csv')
    # data = pd.read_csv('data/gpt_split.csv')
    data = pd.read_csv('data/gpt.csv')
    data = data.sample(frac=0.1, random_state=42)

    X = data.iloc[:,:273]
    y = data['technique']

    # train_index = np.asarray([0, 2, 4, 6, 9, 11, 13])
    # test_index = np.asarray([1, 3, 5, 7, 8, 10, 12])
    # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    # y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    param_C = 1
    param_gamma = 1/X.shape[1]

    # clf = SVM(C=1, gamma=1, verbose=True)
    clf = SVM(C=param_C, gamma=param_gamma, verbose=False)
    print('\nSTART TRAINING MY SVM')
    start_time = time.time()
    clf.fit(X_train, y_train)
    print('FINISHED IN {:.5f} SECONDS'.format(time.time()-start_time))
    y_pred = clf.predict(X_test)
    print('ACCURACY: {:.5f}%'.format(accuracy_score(y_test, y_pred) * 100))

    start_time = time.time()
    clf_2 = SVC(kernel='rbf', C=param_C, gamma=param_gamma, decision_function_shape='ovo')
    print("\nSTART TRAINING SCIKIT's SVM")
    start_time = time.time()
    clf_2.fit(X_train, y_train)
    print('FINISHED IN {:.5f} SECONDS'.format(time.time()-start_time))
    y_pred = clf_2.predict(X_test)
    print("ACCURACY: {:.5f}% ".format(accuracy_score(y_test, y_pred) * 100))


if __name__ == '__main__':
    # main_goa()
    main_svm()
