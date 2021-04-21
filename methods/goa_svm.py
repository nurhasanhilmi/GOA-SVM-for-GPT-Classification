# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:53, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#                                                                                                       %
# Modified by "Ahmad Nurhasan Hilmi" at 02/04/2021                                                      %
# ------------------------------------------------------------------------------------------------------%

import pandas as pd
from numpy import exp, zeros, remainder, clip, sqrt, sum, array, mean
from numpy.random import uniform, normal
from copy import deepcopy

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class GOA_SVM:
    """
    The original version of: Grasshopper Optimization Algorithm (GOA)
        (Grasshopper Optimisation Algorithm: Theory and Application Advances in Engineering Software)
    Link:
        http://dx.doi.org/10.1016/j.advengsoft.2017.01.004
        https://www.mathworks.com/matlabcentral/fileexchange/61421-grasshopper-optimisation-algorithm-goa
    Notes:
        + I added normal() component to Eq, 2.7
        + Changed the way to calculate distance between two location
        + Used batch-size idea
    """
    ID_MAX_PROB = -1  # max problem

    ID_POS = 0  # Position
    ID_FIT = 1  # Fitness

    EPSILON = 10E-10

    iteration = 0

    def __init__(self, k_fold=3, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, c_minmax=(0.00004, 1)):
        self.k_fold = k_fold

        self.verbose = verbose
        self.__check_parameters__(lb, ub)

        self.epoch = epoch
        self.pop_size = pop_size
        self.c_minmax = c_minmax

        self.samples, self.targets = None, None

        self.solution, self.loss_train = None, []

        self.movement = []

    def __check_parameters__(self, lb, ub):
        if (lb is None) or (ub is None):
            print("Lower bound and upper bound are undefined.")
            exit(0)
        else:
            if isinstance(lb, list) and isinstance(ub, list):
                if len(lb) == len(ub):
                    if len(lb) == 0:
                        print("Wrong lower bound and upper bound parameters.")
                        exit(0)
                    else:
                        self.problem_size = len(lb)
                        self.lb = array(lb)
                        self.ub = array(ub)
                else:
                    print("Lower bound and Upper bound need to be same length")
                    exit(0)
            else:
                print("Lower bound and Upper bound need to be a list.")
                exit(0)

    def create_solution(self):
        """ Return the position position with 2 element: position of position and fitness of position
        """
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position)
        return [position, fitness]

    def get_fitness_position(self, position=None, generation=0):
        """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :return:
        """

        kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        X = self.samples
        y = self.targets

        if self.verbose:
            print(
                f'  Grasshopper-{self.iteration+1} Get fitness with pos: {position}')

        scores = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            clf = SVC(kernel='rbf', decision_function_shape='ovo',
                      C=position[0], gamma=position[1])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
            if self.verbose:
                print(f'\tFold-{i+1} : {acc}')
        mean_acc = mean(scores)
        if self.verbose:
            print(f'\tMean Acc: {mean_acc}')
        self.movement.append(
            [generation, self.iteration+1, position[0], position[1], mean_acc])
        self.iteration += 1
        return mean_acc

    def get_global_best_solution(self, pop=None, id_fit=None, id_best=None):
        """ Sort a copy of population and return the copy of the best position """
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        return deepcopy(sorted_pop[id_best])

    def update_global_best_solution(self, pop=None, id_best=None, g_best=None):
        """ Sort the copy of population and update the current best position.
        Return the new current best position
        """
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        return deepcopy(current_best) if current_best[self.ID_FIT] > g_best[self.ID_FIT] else deepcopy(g_best)

    def amend_position(self, position=None):
        return clip(position, self.lb, self.ub)

    def _s_function__(self, r_vector=None):
        # Eq.(2.3) in the paper
        f = 0.5
        l = 1.5
        return f * exp(-r_vector / l) - exp(-r_vector)

    # def manual_create_solution(self):
    #     result = []
    #     C = [1, 10, 100, 1000, 10000]
    #     gamma = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

    #     for i in gamma:
    #         for j in C:
    #             position = [i, j]
    #             fitness = self.get_fitness_position(position=position)
    #             result.append([array(position), fitness])
    #     return result

    def train(self, X, y):
        self.samples = X
        self.targets = y

        if self.verbose:
            print('Generation : 0 (Initialization)')
        pop = [self.create_solution() for _ in range(self.pop_size)]
        # pop = self.manual_create_solution()
        g_best = self.get_global_best_solution(
            pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MAX_PROB)

        if self.verbose:
            print("> Epoch: Init, Best fit: {}, Best pos: {}".format(
                g_best[self.ID_FIT], g_best[self.ID_POS]))
        self.iteration = 0

        for epoch in range(self.epoch):
            if self.verbose:
                print('Generation :', epoch+1)

            # Eq.(2.8) in the paper
            c = self.c_minmax[1] - epoch * \
                ((self.c_minmax[1] - self.c_minmax[0]) / self.epoch)
            for i in range(0, self.pop_size):
                s_i_total = zeros(self.problem_size)
                for j in range(0, self.pop_size):
                    dist = sqrt(
                        sum((pop[i][self.ID_POS] - pop[j][self.ID_POS])**2))

                    # xj - xi / dij in Eq.(2.7)
                    r_ij_vector = (pop[i][self.ID_POS] - pop[j]
                                   [self.ID_POS]) / (dist + self.EPSILON)

                    # |xjd - xid| in Eq. (2.7)
                    xj_xi = 2 + remainder(dist, 2)

                    # The first part inside the big bracket in Eq. (2.7)
                    ran = (c / 2) * (self.ub - self.lb)
                    s_ij = ran * self._s_function__(xj_xi) * r_ij_vector
                    s_i_total += s_ij

                # Eq. (2.7) in the paper
                x_new = c * normal() * s_i_total + g_best[self.ID_POS]
                x_new = self.amend_position(x_new)
                fit = self.get_fitness_position(x_new, generation=epoch+1)
                pop[i] = [x_new, fit]

                if (i + 1) % self.pop_size == 0:
                    g_best = self.update_global_best_solution(
                        pop, self.ID_MAX_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}, Best pos: {}".format(
                    epoch + 1, g_best[self.ID_FIT], g_best[self.ID_POS]))
            self.iteration = 0
        self.solution = g_best
        self.__fit()
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

    def __fit(self):
        param = self.solution[self.ID_POS]

        self.model = SVC(kernel='rbf', decision_function_shape='ovo',
                         C=param[0], gamma=param[1])

        self.min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.min_max_scaler.fit(self.samples)
        X_train = self.min_max_scaler.transform(self.samples)

        self.model.fit(X_train, self.targets)

    def predict(self, X_test, y_test=None):
        X = self.min_max_scaler.transform(X_test)
        y_pred = self.model.predict(X)

        if y_test is not None:
            self.test_samples = pd.DataFrame(X_test)
            self.test_samples['target'] = y_test.tolist()
            self.test_samples['prediction'] = y_pred.tolist()

        return y_pred

    def save_movement_to_csv(self, filename='movements'):
        columns = ['generation', 'grasshopper', 'C', 'gamma', 'fitness']
        df = pd.DataFrame(self.movement, columns=columns)
        name = './data/misc/'+filename+'.csv'
        df.to_csv(name, index=False)
