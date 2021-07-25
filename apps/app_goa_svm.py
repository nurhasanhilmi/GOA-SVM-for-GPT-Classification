import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split

from .constants import RANDOM_STATE, DATASET_PATH
from .goa_svm import GOA_SVM


def app():
    st.title('GOA-SVM')

    st.header('Dataset Setting')
    col1, col2 = st.beta_columns(2)
    with col1:
        sampling_size = st.number_input('Sampling Size (%):', 5, 100, 100, 5)
    with col2:
        train_size = st.number_input('Train Size (%):', 60, 90, 90, 5)

    df = pd.read_csv(DATASET_PATH)
    if sampling_size != 100:
        sampling_size = sampling_size/100
        df = df.sample(frac=sampling_size, random_state=RANDOM_STATE)
    X = df.iloc[:, :156]
    y = df['technique']
    y_sub = df['subtechnique']

    train_size = train_size/100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        random_state=RANDOM_STATE,
        stratify=y_sub
    )
    y_train_sub = y_sub[y_train.index]

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
            filename = 'GOASVM_' + filename

    with col2:
        range_sigma = st.slider("log\u2082\u03c3 Range: ", -5, 15, (0, 10))
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
        best_pos, best_fit = md.train(X_train, y_train, y_train_sub, bar_progress)

        st.write('<hr>', unsafe_allow_html=True)
        st.subheader('Solution :')
        solution_C = f'{2**best_pos[0]:.4f} ({best_pos[0]:.4f})'
        solution_sigma = f'{2**best_pos[1]:.4f} ({best_pos[1]:.4f})'
        model_solution = pd.DataFrame(
            [solution_C, solution_sigma, "{0:.4f}".format(best_fit)],
            index=['Best C (log\u2082C)', 'Best \u03c3 (log\u2082\u03c3)',
                   'Fitness*'],
            columns=['Value']
        )
        st.table(model_solution)
        st.text('*Average of Cross Validation (CV) Matthews Correlation Coefficient (MCC).')

        st.write('<hr>', unsafe_allow_html=True)
        st.subheader('Grasshoppers Movement :')
        mov_columns = ['Iteration', 'Grasshopper', 'C', 'Sigma', 'Fitness']
        mov = pd.DataFrame(md.movement, columns=mov_columns)

        mov['C'] = np.round(mov['C'], decimals=4)
        mov['Sigma'] = np.round(mov['Sigma'], decimals=4)
        mov['Fitness'] = np.round(mov['Fitness'], decimals=4)

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
                "Sigma": "log<sub>2</sub>\u03C3"
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

        mcc = matthews_corrcoef(y_test, y_pred)
        st.subheader(f'MCC: `{mcc:.4f}`')

        if save:
            md.save_movement_to_csv(filename)
            name = './data/misc/'+filename+'.sav'
            pickle.dump(md, open(name, 'wb'))
            st.markdown(
                '<span style="color:blue">*The model has been saved.*</span>', True)
