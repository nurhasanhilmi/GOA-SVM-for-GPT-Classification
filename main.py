import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from methods.goa_svm import GOA_SVM
from methods.svm import SupportVectorClassifier, rbf_kernel

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
    data = pd.read_csv('data/gpt_split_sample.csv')
    X = data.iloc[:,:273]
    y = data['technique']

    train_index = np.asarray([0, 2, 4, 6, 9, 11, 13])
    test_index = np.asarray([1, 3, 5, 7, 8, 10, 12])

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    clf = SupportVectorClassifier(C=1, gamma=1, verbose=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Akurasi: ', accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    main_goa()
    main_svm()
