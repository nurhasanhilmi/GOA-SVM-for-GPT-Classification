import numpy as np
from pandas import factorize
import time


def rbf_kernel(gamma):
    def f(x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f


class SupportVectorClassifier:
    class BaseSVM:
        def __init__(self, C=1.0, gamma=None):
            self.C = C
            self.gamma = gamma
            self.kernel = rbf_kernel
            self.support_ = None
            self.support_vectors_ = None
            self.dual_coef_ = None
            self.intercept_ = None

        def train(self, X, y):
            n_samples, n_features = np.shape(X)
            # initialize alpha array (lagrange multipliers) to all zero
            alpha = np.zeros(n_samples)
            # initialize gradient array to all -1
            gradient = -np.ones(n_samples)
            epsilon = 1e-3  # stopping tolerance
            tau = 1e-12

            # set gamma to 1/(n_features * variance of samples) by default
            if not self.gamma:
                self.gamma = 1 / (n_features * np.var(X))

            # initialize rbf kernel method with parameter gamma
            self.kernel = self.kernel(gamma=self.gamma)

            kernel_matrix = np.zeros((n_samples, n_samples))
            q_matrix = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    kernel_matrix[i, j] = self.kernel(X[i], X[j])
                    q_matrix[i, j] = y[i] * y[j] * kernel_matrix[i, j]

            def select_working_set():
                # select i
                i = -1
                max_gradient = float('-inf')
                min_gradient = float('inf')
                for t in range(n_samples):
                    if (y[t] == +1 and alpha[t] < self.C) or (y[t] == -1 and alpha[t] > 0):
                        if -y[t] * gradient[t] >= max_gradient:
                            i = t
                            max_gradient = -y[t] * gradient[t]

                # select j
                j = -1
                obj_min = float('inf')  # minimal value objective function
                for t in range(n_samples):
                    if (y[t] == +1 and alpha[t] > 0) or (y[t] == -1 and alpha[t] < self.C):
                        b = max_gradient + y[t] * gradient[t]
                        if -y[t] * gradient[t] <= min_gradient:
                            min_gradient = -y[t] * gradient[t]
                        if b > 0:
                            a = q_matrix[i, i] + q_matrix[t, t] - \
                                2 * y[i] * y[t] * q_matrix[i, t]
                            if a <= 0:
                                a = tau
                            if -(b * b) / a <= obj_min:
                                j = t
                                obj_min = -(b * b) / a

                if max_gradient - min_gradient < epsilon:
                    return -1, -1
                return i, j

            while True:
                i, j = select_working_set()
                if j == -1:
                    break

                # working set is (i, j)
                a = q_matrix[i, i] + q_matrix[j, j] - \
                    2 * y[i] * y[j] * q_matrix[i, j]
                if a <= 0:
                    a = tau
                b = -y[i] * gradient[i] + y[j] * gradient[j]

                # update alpha
                old_alpha_i = alpha[i]
                old_alpha_j = alpha[j]
                alpha[i] += y[i] * b / a
                alpha[j] -= y[j] * b / a

                # project alpha back to the feasible region
                sum_value = y[i] * old_alpha_i + y[j] * old_alpha_j
                if alpha[i] > self.C:
                    alpha[i] = self.C
                elif alpha[i] < 0:
                    alpha[i] = 0

                alpha[j] = y[j] * (sum_value - y[i] * alpha[i])
                if alpha[j] > self.C:
                    alpha[j] = self.C
                elif alpha[j] < 0:
                    alpha[j] = 0

                alpha[i] = y[i] * (sum_value - y[j] * alpha[j])

                # update gradient
                delta_alpha_i = alpha[i] - old_alpha_i
                delta_alpha_j = alpha[j] - old_alpha_j
                for t in range(n_samples):
                    gradient[t] += q_matrix[t, i] * delta_alpha_i + \
                        q_matrix[t, j] * delta_alpha_j

            def calculate_intercept():
                num_free = 0
                upper_bound = float('inf')
                lower_bound = float('-inf')
                sum_free = 0
                for i in range(n_samples):
                    y_g = y[i] * gradient[i]

                    if alpha[i] >= self.C:
                        if y[i] == -1:
                            upper_bound = np.minimum(upper_bound, y_g)
                        else:
                            lower_bound = np.maximum(lower_bound, y_g)
                    elif alpha[i] <= 0:
                        if y[i] == +1:
                            upper_bound = np.minimum(upper_bound, y_g)
                        else:
                            lower_bound = np.maximum(lower_bound, y_g)
                    else:
                        num_free += 1
                        sum_free += y_g
                if num_free > 0:
                    intercept = sum_free / num_free
                else:
                    intercept = (upper_bound + lower_bound) / 2
                return -intercept

            self.support_ = np.nonzero(alpha)[0]
            self.dual_coef_ = y[self.support_] * alpha[self.support_]
            self.support_vectors_ = X[self.support_]
            self.intercept_ = calculate_intercept()

        def decision_function(self, X):
            result = np.zeros(np.shape(X)[0])
            for i, sample in enumerate(X):
                for j in range(len(self.support_vectors_)):
                    result[i] += self.dual_coef_[j] * \
                        self.kernel(self.support_vectors_[j], sample)
            result += self.intercept_
            return result

        def test(self, X):
            return np.sign(self.decision_function(X))

    def __init__(self, C=1.0, gamma=None, verbose=False):
        self.C = C
        self.gamma = gamma

        self.classifiers = []
        self.target_labels = []
        self.labels = None

        self.verbose = verbose

        # self.support_ = None
        # self.support_vectors_ = None
        # self.dual_coef_ = None
        # self.intercept_ = None

    def fit(self, X, y):
        y, self.labels = factorize(y)
        n_classes = len(self.labels)

        idx = 0
        for i in range(n_classes - 1):
            for j in range(i + 1, n_classes):
                self.target_labels.append(self.labels[[i, j]])
                sample = X[(y == i) | (y == j)]
                target = y[(y == i) | (y == j)]
                target[target == i] = -1
                target[target == j] = 1

                start_time = time.time()
                if self.verbose:
                    print(idx + 1)
                    print(
                        f'Start training for \t{np.array(self.labels[[i, j]])}\t{len(target[target == -1])}\t{len(target[target == 1])}')
                self.classifiers.append(
                    self.BaseSVM(C=self.C, gamma=self.gamma))
                self.classifiers[idx].train(sample, target)
                if self.verbose:
                    print(
                        f'Finish training for \t{self.target_labels[idx]}\t{time.time() - start_time} seconds')
                idx += 1
        self.target_labels = np.array(self.target_labels)

    def predict(self, X):
        y_pred = []

        for sample in X:
            votes = []
            for i, classifier in enumerate(self.classifiers):
                prediction = classifier.test([sample])
                if prediction == -1:
                    votes.append(self.target_labels[i][0])
                else:
                    votes.append(self.target_labels[i][1])
            y_pred.append(max(self.labels, key=votes.count))

        return np.array(y_pred)
