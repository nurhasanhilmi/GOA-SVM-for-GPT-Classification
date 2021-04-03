import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from methods.goa_svm import Goa_Svm


def main():
    data = pd.read_csv('data/gpt.csv')

    x = data.iloc[:,:273]
    y = data['technique']

    verbose = True
    lb = [2.27126176e+02, 1.82486880e-02]
    ub = [6.96697373e+03, 4.15438806e-02]
    pop_size = 30
    epoch = 10
    md = Goa_Svm(lb=lb, ub=ub, verbose=verbose, pop_size=pop_size, epoch=epoch)
    best_pos, best_fit, _ = md.train(x, y)
    print(best_pos)
    print(best_fit)


if __name__ == '__main__':
    main()