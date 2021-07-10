import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class GridSearchSVM:
    def __init__(self, solution, k_fold, X_train, X_test, y_train, y_test, y_pred, model, min_max_scaler, lb, ub, step_size):
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
        self.step_size = step_size


def app():
    st.title('Grid Search-SVM')

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

    st.header('Paremeter Setting')
    C = []
    sigma = []
    C_header = []
    sigma_header = []

    col1, col2 = st.beta_columns(2)
    with col1:
        range_C = st.slider("log\u2082C Set:", -5, 15, (0, 10))

    with col2:
        range_sigma = st.slider("log\u2082\u03c3 Set:", -5, 15, (0, 7))

    step_size = st.number_input('Step Size:', 0.1, 1.0, 0.5, 0.05)
    for c in np.round(np.arange(range_C[0], range_C[1]+0.001, step_size), decimals=2):
        C.append(2**c)
        C_header.append(f'2^{c}')
    for g in np.round(np.arange(range_sigma[0], range_sigma[1]+0.01, step_size), decimals=2):
        sigma.append(2**g)
        sigma_header.append(f'2^{g}')

    col3, col4 = st.beta_columns(2)
    with col3:
        C_df = pd.DataFrame(C)
        C_df = C_df.transpose()
        C_df.columns = C_header
        C_df.index = ['value']
        st.write('C Set:')
        st.dataframe(C_df)
    with col4:
        sigma_df = pd.DataFrame(sigma)
        sigma_df = sigma_df.transpose()
        sigma_df.columns = sigma_header
        sigma_df.index = ['value']
        st.write('\u03c3 Set:')
        st.dataframe(sigma_df)

    lb = (range_C[0], range_sigma[0])
    ub = (range_C[1], range_sigma[1])

    k_fold = st.number_input('K-Fold :', 2, 10, 10, step=1)
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
        best_sigma = None
        best_C = None
        pop = np.zeros((len(C)*len(sigma), 3))
        iter_progress = 0
        for i, c in enumerate(C):
            for j, g in enumerate(sigma):
                if verbose:
                    print(f'C: {c}, Sigma: {g}')
                scores = []
                for k, (train_index, test_index) in enumerate(kf.split(X)):

                    X_trn, X_tst = X.iloc[train_index], X.iloc[test_index]
                    y_trn, y_tst = y.iloc[train_index], y.iloc[test_index]

                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaler.fit(X_trn)
                    X_trn = scaler.transform(X_trn)
                    X_tst = scaler.transform(X_tst)

                    gamma_value = 1/(2*g**2)
                    clf = SVC(kernel='rbf',
                              decision_function_shape='ovo', C=c, gamma=gamma_value)
                    clf.fit(X_trn, y_trn)
                    y_pred = clf.predict(X_tst)

                    mcc = matthews_corrcoef(y_tst, y_pred)
                    scores.append(mcc)
                    if verbose:
                        print(f'\tFold-{k+1} : {mcc}')
                avg_mcc = np.mean(scores)
                pop[iter_progress] = ['%.1f' %
                                      np.log2(c), '%.1f' % np.log2(g), avg_mcc]
                if avg_mcc > best_score:
                    best_score = avg_mcc
                    best_sigma = g
                    best_C = c
                if verbose:
                    print('   Fitness:', avg_mcc)
                iter_progress += 1
                bar_progress.progress(iter_progress/(len(C)*len(sigma)))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X_train)
        X_train_ = scaler.transform(X_train)
        X_test_ = scaler.transform(X_test)
        best_gamma_value = 1/(2*best_sigma**2)
        clf = SVC(kernel='rbf', decision_function_shape='ovo',
                  C=best_C, gamma=best_gamma_value)
        clf.fit(X_train_, y_train)

        if verbose:
            print(
                f'Best Parameter (C, Sigma) : ({best_C}, {best_sigma}). Fitness: {best_score}')

        st.subheader('Solution :')
        solution_C = f'{best_C:.4f} ({np.log2(best_C):.1f})'
        solution_sigma = f'{best_sigma:.4f} ({np.log2(best_sigma):.1f})'
        model_solution = pd.DataFrame(
            [solution_C, solution_sigma, "{0:.4f}".format(best_score)],
            index=['Best C (log\u2082C)', 'Best \u03c3 (log\u2082\u03c3)',
                   'Fitness'],
            columns=['Value']
        )
        st.table(model_solution)

        columns = ['C', 'Sigma', 'Fitness']
        pop_df = pd.DataFrame(pop, columns=columns)
        movement = pop_df.pivot(
            index='C', columns='Sigma', values='Fitness')
        fig = px.imshow(
            np.asmatrix(movement),
            labels=dict(x="log\u2082\u03c3", y="log\u2082C",
                        color="Fitness"),
            x=[str(x) for x in movement.columns],
            y=[str(x) for x in movement.index],
            width=600,
            height=600
        )
        # fig.update_xaxes(side="top")
        st.write(fig)

        st.subheader('Evaluate The Model')
        y_pred = clf.predict(X_test_)

        test_sample = pd.DataFrame(X_test)
        test_sample['target'] = y_test
        test_sample['prediction'] = y_pred
        st.write('Test Samples + Prediction')
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

        mcc = matthews_corrcoef(y_test, y_pred)
        st.subheader(f'MCC: `{mcc:.4f}`')

        if save:
            name = './data/misc/'+filename+'.csv'
            pop_df.to_csv(name, index=False)

            best_param = [best_C, best_sigma]
            solution = [best_param, best_score]
            md = GridSearchSVM(solution, k_fold, X_train, X_test,
                               y_train, y_test, y_pred, clf, scaler, lb, ub, step_size)
            name = './data/misc/'+filename+'.sav'
            pickle.dump(md, open(name, 'wb'))
            st.markdown(
                '<span style="color:blue">*The model has been saved.*</span>', True)
