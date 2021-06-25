from numpy import exp, zeros, remainder, clip, sqrt, sum, array
from numpy.random import uniform, normal
from copy import deepcopy


class BaseGOA:
    ID_MAX_PROB = -1  # max problem

    ID_POS = 0  # Position
    ID_FIT = 1  # Fitness

    EPSILON = 10E-10

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=100, pop_size=50, c_minmax=(0.00004, 1)):
        self.verbose = verbose
        self.obj_func = obj_func
        self.__check_parameters__(lb, ub)

        self.epoch = epoch
        self.pop_size = pop_size
        self.c_minmax = c_minmax

        self.solution, self.loss_train = None, []

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

    def get_fitness_position(self, position=None):
        """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :return:
        """
        return self.obj_func(position)

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

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MAX_PROB)

        for epoch in range(self.epoch):
            # Eq.(2.8) in the paper
            c = self.c_minmax[1] - epoch * ((self.c_minmax[1] - self.c_minmax[0]) / self.epoch)
            for i in range(0, self.pop_size):
                s_i_total = zeros(self.problem_size)
                for j in range(0, self.pop_size):
                    dist = sqrt(sum((pop[i][self.ID_POS] - pop[j][self.ID_POS])**2))

                    # xj - xi / dij in Eq.(2.7)
                    r_ij_vector = (pop[i][self.ID_POS] - pop[j][self.ID_POS]) / (dist + self.EPSILON)

                    # |xjd - xid| in Eq. (2.7)
                    xj_xi = 2 + remainder(dist, 2)

                    # The first part inside the big bracket in Eq. (2.7)
                    ran = (c / 2) * (self.ub - self.lb)
                    s_ij = ran * self._s_function__(xj_xi) * r_ij_vector
                    s_i_total += s_ij

                x_new = c * normal() * s_i_total + g_best[self.ID_POS]     # Eq. (2.7) in the paper
                x_new = self.amend_position(x_new)
                fit = self.get_fitness_position(x_new)
                pop[i] = [x_new, fit]

                if (i + 1) % self.pop_size == 0:
                    g_best = self.update_global_best_solution(pop, self.ID_MAX_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
