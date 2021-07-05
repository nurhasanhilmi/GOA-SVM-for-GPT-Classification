import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class GridSearchSVM:
    def __init__(self, solution, k_fold, X_train, X_test, y_train, y_test, y_pred, model, min_max_scaler, lb, ub):
        self.solution = solution
        self.k_fold = k_fold
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.model = model
        self.min_max_scaler = min_max_scaler
        self.lb = lb
        self.ub = ub


def app():
    st.title('Grid Search-SVM')

    st.header('Dataset Setting')
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        selected_dataset = st.selectbox(
            'Select Dataset:', ['GPT Complete', 'GPT Split'])
    with col2:
        train_size = st.number_input('Train Size (%):', 60, 90, 80, step=5)
    with col3:
        sampling_size = st.number_input(
            'Sampling Size (%):', 5, 100, 10, step=5)

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

    st.header('Paremeter Setting')
    range_C = st.slider("C Set (Exponents of 2):", -20, 20, (0, 10), 1)
    range_gamma = st.slider(
        "Gamma Set (Exponents of 2) :", -20, 20, (-14, -1), 1)

    lb = (range_C[0], range_gamma[0])
    ub = (range_C[1], range_gamma[1])
    C = []
    gamma = []
    # C_header = []
    # gamma_header = []
    for c in range(int(range_C[0]), int(range_C[1]+1)):
        C.append(2**c)
        # C_header.append(f'2^{c}')

    for g in range(int(range_gamma[0]), int(range_gamma[1])+1):
        gamma.append(2**g)
        # gamma_header.append(f'2^{g}')

    # C_df = pd.DataFrame(C)
    # C_df = C_df.transpose()
    # C_df.columns = C_header
    # st.write('C range value:')
    # st.dataframe(C_df)

    # gamma_df = pd.DataFrame(gamma)
    # gamma_df = gamma_df.transpose()
    # gamma_df.columns = gamma_header
    # st.write('Gamma range value:')
    # st.dataframe(gamma_df)

    k_fold = st.number_input('K-Fold :', 2, 10, 5, step=1)
    verbose = st.checkbox('Show Backend Output (Verbose)', value=True)
    save = st.checkbox('Save Model')
    if save:
        filename = st.text_input('Filename:')
        filename = selected_dataset + '_GridSearchSVM_' + filename

    if st.button('Train & Test'):
        st.write('**Start The Training Process**')
        bar_progress = st.progress(0.0)

        kf = KFold(n_splits=k_fold, shuffle=True, random_state=0)
        X = X_train
        y = y_train

        best_score = 0
        best_gamma = None
        best_C = None
        pop = np.zeros((len(C)*len(gamma), 3))
        iter_progress = 0
        for i, c in enumerate(C):
            for j, g in enumerate(gamma):
                if verbose:
                    print(f'C: {c}, Gamma: {g}')
                scores = []
                for k, (train_index, test_index) in enumerate(kf.split(X)):

                    X_trn, X_tst = X.iloc[train_index], X.iloc[test_index]
                    y_trn, y_tst = y.iloc[train_index], y.iloc[test_index]

                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaler.fit(X_trn)
                    X_trn = scaler.transform(X_trn)
                    X_tst = scaler.transform(X_tst)

                    clf = SVC(kernel='rbf',
                              decision_function_shape='ovo', C=c, gamma=g)
                    clf.fit(X_trn, y_trn)
                    y_pred = clf.predict(X_tst)

                    f1score = f1_score(y_tst, y_pred, average='weighted')
                    scores.append(f1score)
                    if verbose:
                        print(f'\tFold-{k+1} : {f1score}')
                avg_fscore = np.mean(scores)
                pop[iter_progress] = [c, g, avg_fscore]
                if avg_fscore > best_score:
                    best_score = avg_fscore
                    best_gamma = g
                    best_C = c
                if verbose:
                    print('\tAvg. F-Score:', avg_fscore)
                iter_progress += 1
                bar_progress.progress(iter_progress/(len(C)*len(gamma)))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X_train)
        X_train_ = scaler.transform(X_train)
        X_test_ = scaler.transform(X_test)
        clf = SVC(kernel='rbf', decision_function_shape='ovo',
                  C=best_C, gamma=best_gamma)
        clf.fit(X_train_, y_train)

        if verbose:
            print(
                f'Best Parameter (C, Gamma) : ({best_C}, {best_gamma}). Fitness: {best_score}')

        st.write('***Solution :***')
        sbest_C = f'{best_C} (2^{int(np.log2(best_C))})'
        sbest_gamma = f'{best_gamma} (2^{int(np.log2(best_gamma))})'
        model_solution = pd.DataFrame(
            [sbest_C, sbest_gamma, "{0:.2%}".format(best_score)],
            index=['Best C', 'Best Gamma',
                   'Avg. F-Score'],
            columns=['Value']
        )
        st.table(model_solution)

        columns = ['C', 'gamma', 'fitness']
        pop_df = pd.DataFrame(pop, columns=columns)
        movement = pop_df.pivot(index='C', columns='gamma', values='fitness')
        fig = px.imshow(
            np.asmatrix(movement),
            labels=dict(x="Gamma", y="C", color="Avg. Val. F-Score"),
            x=[str(x) for x in movement.columns],
            y=[str(x) for x in movement.index],
            width=700,
            height=700,
            color_continuous_scale='RdBu_r'
        )
        fig.update_xaxes(side="top")
        st.write(fig)

        st.write('**Evaluate The Model**')
        y_pred = clf.predict(X_test_)

        test_sample = pd.DataFrame(X_test)
        test_sample['target'] = y_test
        test_sample['prediction'] = y_pred
        st.write(f'Test Samples + Prediction')
        st.dataframe(test_sample)

        fig = plt.figure()
        conf_matrix = confusion_matrix(y_test, y_pred)
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
        st.write('**F-score: **', f1_score(y_test, y_pred, average='weighted'))

        if save:
            name = './data/misc/'+filename+'.csv'
            pop_df.to_csv(name, index=False)

            best_param = [best_C, best_gamma]
            solution = [best_param, best_score]
            md = GridSearchSVM(solution, k_fold, X_train, X_test,
                               y_train, y_test, y_pred, clf, scaler, lb, ub)
            name = './data/misc/'+filename+'.sav'
            pickle.dump(md, open(name, 'wb'))
            st.markdown(
                '<span style="color:blue">*The model has been saved.*</span>', True)
