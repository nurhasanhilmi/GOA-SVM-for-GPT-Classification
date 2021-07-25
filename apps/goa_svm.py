from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from .constants import RANDOM_STATE


class GOA_SVM:
    ID_POS = 0  # Position
    ID_FIT = 1  # Fitness

    EPSILON = 10E-10
    iteration = 0

    def __init__(self, k_fold=5, lb=None, ub=None, verbose=True, epoch=10, pop_size=30, c_minmax=(0.00004, 1)):
        self.k_fold = k_fold
        self.verbose = verbose

        self.problem_size = len(lb)
        self.lb = np.array(lb)
        self.ub = np.array(ub)

        self.epoch = epoch
        self.pop_size = pop_size
        self.c_minmax = c_minmax

        self.samples, self.targets = None, None
        self.solution = None
        self.movement = []

        self.scaler = MinMaxScaler(feature_range=(-1,1))
    
    def init_solution(self, i, progress):
        position = np.random.uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position)
        progress.progress((i+1)/(self.epoch*self.pop_size))
        return [position, fitness]

    def get_fitness_position(self, position=None, generation=1):
        skf = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=RANDOM_STATE)

        X = np.asarray(self.samples)
        y = np.asarray(self.targets)
        y_sub = np.asarray(self.targets_sub)

        if self.verbose:
            print(f'   I_{self.iteration+1}^{generation} POS: {position}')

        scores = []
        for i, (train_index, val_index) in enumerate(skf.split(X, y_sub)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_val = self.scaler.transform(X_val)

            sigma_value = 2**position[1]
            gamma_value = 1/(2*sigma_value**2)
            C_value = 2**position[0]
            clf = SVC(kernel='rbf', decision_function_shape='ovo', C=C_value, gamma=gamma_value)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            mcc = matthews_corrcoef(y_val, y_pred)
            scores.append(mcc)
            if self.verbose: print(f'\tFold-{i+1} : {mcc}')
        avg_mcc = np.mean(scores)
        if self.verbose: print(f'\tFIT: {avg_mcc}')
        self.movement.append([generation, self.iteration+1, position[0], position[1], avg_mcc])
        self.iteration += 1
        return avg_mcc

    def get_global_best_solution(self, pop=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        return deepcopy(sorted_pop[-1])

    def update_global_best_solution(self, pop=None, g_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[-1]
        return deepcopy(current_best) if current_best[self.ID_FIT] > g_best[self.ID_FIT] else deepcopy(g_best)

    def amend_position(self, position=None):
        return np.clip(position, self.lb, self.ub)

    def _S_func(self, r=None):
        # Eq.(2.3) in the paper
        f = 0.5
        l = 1.5
        return f * np.exp(-r / l) - np.exp(-r)

    def train(self, X, y, y_sub, progress):
        self.samples = X
        self.targets = y
        self.targets_sub = y_sub

        if self.verbose:
            print('\n> BEGIN Iteration : 1 (Initialization)')

        pop = [self.init_solution(i, progress) for i in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop)

        if self.verbose:
            print("> END Iteration: 1 (Initialization), Best fit: {}, Best pos: {}".format(
                g_best[self.ID_FIT], g_best[self.ID_POS]))
        self.iteration = 0

        for epoch in range(2, self.epoch+1):
            if self.verbose:
                print('\n> BEGIN Iteration :', epoch)

            # UPDATE COEFFICIENT c
            # Eq.(2.8) in the paper
            c = self.c_minmax[1] - epoch * \
                ((self.c_minmax[1] - self.c_minmax[0]) / self.epoch)

            # UPDATE POSITION OF EACH GRASSHOPPER
            for i in range(self.pop_size):
                temp = pop
                s_i_total = np.zeros(self.problem_size)
                for j in range(self.pop_size):
                    if i != j:
                        # Calculate the distance between two grasshoppers
                        # dist = np.sqrt(sum((pop[j][self.ID_POS] - pop[i][self.ID_POS])**2))
                        dist = np.linalg.norm(
                            pop[j][self.ID_POS] - pop[i][self.ID_POS])
                        # xj-xi/dij in Eq. (2.7)
                        r_ij = (pop[j][self.ID_POS] - pop[i]
                                [self.ID_POS]) / (dist + self.EPSILON)
                        # |xjd - xid| in Eq. (2.7)
                        xj_xi = 2 + np.remainder(dist, 2)
                        # The first part inside the big bracket in Eq. (2.7)
                        s_ij = ((self.ub - self.lb)*c/2) * \
                            self._S_func(xj_xi) * r_ij
                        s_i_total += s_ij
                # Eq. (2.7) in the paper
                x_new = c * s_i_total + g_best[self.ID_POS]
                # Relocate grasshoppers that go outside the search space
                temp[i][self.ID_POS] = self.amend_position(x_new)
            pop = temp
            # CALCULATE FITNESS OF EACH GRASSHOPPER
            for i in range(self.pop_size):
                fit = self.get_fitness_position(
                    pop[i][self.ID_POS], generation=epoch)
                pop[i][self.ID_FIT] = fit
                progress.progress(((epoch-1)*self.pop_size+i+1) /
                                  (self.epoch*self.pop_size))

            # UPDATE T IF THERE IS A BETTER SOLUTION
            g_best = self.update_global_best_solution(pop, g_best)

            if self.verbose:
                print("> END Iteration: {}, Best fit: {}, Best pos: {}".format(
                    epoch, g_best[self.ID_FIT], g_best[self.ID_POS]))
            self.iteration = 0

        self.solution = g_best
        self.fit()
        return g_best[self.ID_POS], g_best[self.ID_FIT]

    def fit(self):
        param = self.solution[self.ID_POS]

        C_value = 2**param[0]
        sigma_value = 2**param[1]
        gamma_value = 1/(2*sigma_value**2)

        self.model = SVC(kernel='rbf', decision_function_shape='ovo', C=C_value, gamma=gamma_value)

        self.scaler.fit(self.samples)
        X_train = self.scaler.transform(self.samples)
        self.model.fit(X_train, self.targets)

    def predict(self, X_test, y_test=None):
        X = self.scaler.transform(X_test)
        y_pred = self.model.predict(X)
        if y_test is not None:
            self.test_samples = pd.DataFrame(X_test)
            self.test_samples['target'] = y_test.tolist()
            self.test_samples['prediction'] = y_pred.tolist()
        return y_pred

    def save_movement_to_csv(self, filename='movements'):
        columns = ['Iteration', 'Grasshopper', 'C', 'Sigma', 'Fitness']
        df_movement = pd.DataFrame(self.movement, columns=columns)
        name = './data/misc/'+filename+'.csv'
        df_movement.to_csv(name, index=False)
