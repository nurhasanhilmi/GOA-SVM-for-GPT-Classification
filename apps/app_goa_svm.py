import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class GOA_SVM:
    ID_MAX_PROB = -1  # max problem
    ID_POS = 0  # Position
    ID_FIT = 1  # Fitness
    EPSILON = 10E-10
    iteration = 0

    def __init__(self, k_fold=5, lb=None, ub=None, verbose=True, epoch=10, pop_size=30, c_minmax=(0.00004, 1)):
        self.k_fold = k_fold
        self.verbose = verbose
        self.__check_parameters__(lb, ub)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c_minmax = c_minmax
        self.samples, self.targets = None, None
        self.solution = None
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
                        self.lb = np.array(lb)
                        self.ub = np.array(ub)
                else:
                    print("Lower bound and Upper bound need to be same length")
                    exit(0)
            else:
                print("Lower bound and Upper bound need to be a list.")
                exit(0)

    def init_solution(self, i, progress):
        position = np.random.uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position)
        progress.progress((i+1)/(self.epoch*self.pop_size))
        return [position, fitness]

    def get_fitness_position(self, position=None, generation=1):
        kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=0)
        X = self.samples
        y = self.targets

        if self.verbose:
            print(
                f'   I_{self.iteration+1}^{generation} Position: {position}')

        scores = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            sigma_value = 2**position[1]
            gamma_value = 1/(2*sigma_value**2)

            C_value = 2**position[0]
            clf = SVC(kernel='rbf', decision_function_shape='ovo',
                      C=C_value, gamma=gamma_value)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            f1score = f1_score(y_test, y_pred, average='weighted')
            scores.append(f1score)
            if self.verbose:
                print(f'      Fold-{i+1} : {f1score}')
        avg_fscore = np.mean(scores)
        if self.verbose:
            print(f'      Fitness : {avg_fscore}')
        self.movement.append(
            [generation, self.iteration+1, position[0], position[1], avg_fscore])
        self.iteration += 1
        return avg_fscore

    def get_global_best_solution(self, pop=None, id_fit=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
        return deepcopy(sorted_pop[id_best])

    def update_global_best_solution(self, pop=None, id_best=None, g_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        return deepcopy(current_best) if current_best[self.ID_FIT] > g_best[self.ID_FIT] else deepcopy(g_best)

    def amend_position(self, position=None):
        return np.clip(position, self.lb, self.ub)

    def _S_func(self, r=None):
        # Eq.(2.3) in the paper
        f = 0.5
        l = 1.5
        return f * np.exp(-r / l) - np.exp(-r)

    def train(self, X, y, progress):
        self.samples = X
        self.targets = y
        if self.verbose:
            print('\nBEGIN Iteration : 1 (Initialization)')

        pop = [self.init_solution(i, progress) for i in range(self.pop_size)]
        g_best = self.get_global_best_solution(
            pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MAX_PROB)

        if self.verbose:
            print("> END Iteration: 1 (Initialization), Best fit: {}, Best pos: {}".format(
                g_best[self.ID_FIT], g_best[self.ID_POS]))
        self.iteration = 0

        for epoch in range(2, self.epoch+1):
            if self.verbose:
                print('\nBEGIN Iteration :', epoch)

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
            g_best = self.update_global_best_solution(
                pop, self.ID_MAX_PROB, g_best)

            if self.verbose:
                print("> END Iteration: {}, Best fit: {}, Best pos: {}".format(
                    epoch, g_best[self.ID_FIT], g_best[self.ID_POS]))
            self.iteration = 0

        self.solution = g_best
        self.__fit()
        return g_best[self.ID_POS], g_best[self.ID_FIT]

    def __fit(self):
        param = self.solution[self.ID_POS]
        sigma_value = 2**param[1]
        gamma_value = 1/(2*sigma_value**2)
        C_value = 2**param[0]
        self.model = SVC(
            kernel='rbf', decision_function_shape='ovo', C=C_value, gamma=gamma_value)
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
        columns = ['Iteration', 'Grasshopper', 'C', 'Sigma', 'Fitness']
        df_movement = pd.DataFrame(self.movement, columns=columns)
        name = './data/misc/'+filename+'.csv'
        df_movement.to_csv(name, index=False)


def app():
    st.title('GOA-SVM')
    # st.write("Optimization of SVM Parameters using Grasshopper Optimization Algorithm (GOA) for Guitar Playing Technique Classification.")

    st.header('Dataset Setting')
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        selected_dataset = st.selectbox(
            'Select Dataset:', ['GPT Complete', 'GPT Split'])
    with col2:
        train_size = st.number_input('Train Size (%):', 60, 90, 90, step=5)
    with col3:
        sampling_size = st.number_input(
            'Sampling Size (%):', 5, 100, 100, step=5)

    train_size = train_size/100
    sampling_size = sampling_size/100

    if selected_dataset == 'GPT Complete':
        df = pd.read_csv('data/gpt.csv')
        df = df.sample(frac=sampling_size, random_state=0)
        X = df.iloc[:, :273]
        y = df['technique']
    elif selected_dataset == 'GPT Split':
        df = pd.read_csv('data/gpt_split.csv')
        df = df.sample(frac=sampling_size, random_state=0)
        X = df.iloc[:, :273]
        y = df['technique']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=0)
    dataset_summarize = pd.DataFrame(
        [[X_train.shape[1], X_train.shape[0], X_test.shape[0], X.shape[0]]],
        columns=['Num. Features', 'Num. Train Samples',
                 'Num. Test Samples', 'Total Samples']
    )
    st.table(dataset_summarize.assign(hack='').set_index('hack'))

    st.header('Parameter Setting')
    k_fold = st.number_input('K-Fold :', 2, 10, 5, step=1)

    col1, col2 = st.beta_columns(2)
    with col1:
        range_C = st.slider("log\u2082C Range:", -5, 15, (0, 10))

        pop_size = st.number_input('Population Size :', 1, 100, 30, step=5)
        c_min = st.number_input('c_min :', 1e-05, 1.0,
                                4e-05, step=1e-05, format="%.5f")
        verbose = st.checkbox('Show Backend Output (Verbose)', value=True)
        save = st.checkbox('Save Model')
        if save:
            filename = st.text_input('Filename:')
            filename = selected_dataset + '_GOASVM_' + filename

    with col2:
        range_sigma = st.slider("log\u2082\u03c3 Range: ", -5, 15, (0, 5))

        epoch = st.number_input('Maximum Iterations :', 2, 100, 10, step=2)
        c_max = st.number_input('c_max :', 1, 10, 1)

    lb = [range_C[0], range_sigma[0]]
    ub = [range_C[1], range_sigma[1]]
    c_minmax = (c_min, c_max)
    # st.write(k_fold, lb, ub, pop_size, epoch, c_minmax)

    if st.button('Train & Test'):
        st.write('**Start The Training Process**')
        bar_progress = st.progress(0.0)
        md = GOA_SVM(k_fold=k_fold, lb=lb, ub=ub, verbose=verbose,
                     pop_size=pop_size, epoch=epoch, c_minmax=c_minmax)
        best_pos, best_fit = md.train(X_train, y_train, bar_progress)

        st.write('<hr>', unsafe_allow_html=True)
        st.subheader('Solution :')
        solution_C = f'{2**best_pos[0]:.4f} ({best_pos[0]:.4f})'
        solution_sigma = f'{2**best_pos[1]:.4f} ({best_pos[1]:.4f})'
        model_solution = pd.DataFrame(
            [solution_C, solution_sigma, "{0:.2%}".format(best_fit)],
            index=['Best C (log\u2082C)', 'Best \u03c3 (log\u2082\u03c3)',
                   'Fitness*'],
            columns=['Value']
        )
        st.table(model_solution)
        st.text('*Mean CV score of the weighted-average-F1-score of each class.')

        st.write('<hr>', unsafe_allow_html=True)
        st.subheader('Grasshoppers Movement :')
        mov_columns = ['Iteration', 'Grasshopper', 'C', 'Sigma', 'Fitness']
        mov = pd.DataFrame(md.movement, columns=mov_columns)

        mov['C'] = np.round(mov['C'], decimals=4)
        mov['Sigma'] = np.round(mov['Sigma'], decimals=4)
        mov['Fitness'] = np.round(mov['Fitness']*100, decimals=4)

        fig = px.scatter_3d(
            mov,
            x='C',
            y='Sigma',
            z='Fitness',
            color='Iteration',
            width=600,
            height=600,
            labels={
                "C": "log<sub>2</sub>C",
                "Sigma": "log<sub>2</sub>\u03C3",
                "Fitness": "Fitness (%)",
            }
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=range_C,),
                yaxis=dict(range=range_sigma,),
            )
        )
        st.write(fig)

        st.write('<hr>', unsafe_allow_html=True)
        st.subheader('Evaluate The Model')
        y_pred = md.predict(X_test, y_test)

        test_sample = pd.DataFrame(X_test)
        test_sample['target'] = y_test
        test_sample['prediction'] = y_pred
        st.write(f'Test Samples + Prediction')
        st.dataframe(test_sample)

        fig = plt.figure()
        conf_matrix = confusion_matrix(y_test, y_pred)
        print('\nConfusion matrix')
        print(conf_matrix)

        sns.heatmap(
            conf_matrix,
            cmap=sns.color_palette("light:b", as_cmap=True),
            cbar=False,
            annot=True,
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test),
            fmt="d"
        )
        plt.ylabel("Actual", fontweight='bold')
        plt.xlabel("Predicted", fontweight='bold')
        st.pyplot(fig)

        f1score = f1_score(y_test, y_pred, average='weighted')
        print('\nEvalute the model')
        print(classification_report(y_test, y_pred))
        st.subheader(f'Weighted-F1-Score: `{f1score*100:.2f}%`')

        if save:
            md.save_movement_to_csv(filename)
            name = './data/misc/'+filename+'.sav'
            pickle.dump(md, open(name, 'wb'))
            st.markdown(
                '<span style="color:blue">*The model has been saved.*</span>', True)
