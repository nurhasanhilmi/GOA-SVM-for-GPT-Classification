import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def app():
    path = './data/misc/'
    st.title('Load Saved Models')

    list_file = os.listdir(path)
    list_file = [filename[:-4] for filename in list_file]
    list_file.sort()
    filename = st.selectbox('Select Model:', list(dict.fromkeys(list_file)))
    used_dataset = filename.split('_')[0]
    optimizer = filename.split('_')[1]

    if filename:
        movement = pd.read_csv(path+filename+'.csv')
        model = open(path+filename+'.sav', 'rb')
        model = pickle.load(model)

        col1, col2 = st.beta_columns(2)

        with col1:
            st.header('Parameters')
            if optimizer == 'GOASVM':
                epoch = model.epoch
                pop_size = model.pop_size
                c_minmax = model.c_minmax
            k_fold = model.k_fold
            c_range = (model.lb[0], model.ub[0])
            gamma_range = (model.lb[1], model.ub[1])
            if optimizer == 'GOASVM':
                model_parameter = pd.DataFrame(
                    [k_fold, pop_size, epoch, c_minmax, c_range, gamma_range],
                    index=['K-Fold', 'Pop. Size', 'Maximum Iteration',
                        'c minmax', 'C Range', 'Gamma Range'],
                    columns=['Value']
                )
            elif optimizer == 'GridSearchSVM':
                model_parameter = pd.DataFrame(
                    [k_fold , f'[{c_range[0]},...,{c_range[1]}]', f'[{gamma_range[0]},...,{gamma_range[1]}]'],
                    index=['K-Fold', 'C Set (Exp. of 2)', 'Gamma Set (Exp. of 2)'],
                    columns=['Value']
                )
            st.table(model_parameter)
        with col2:
            st.header('Solution')
            best_pos = model.solution[0]
            best_fit = model.solution[1]
            if optimizer == 'GOASVM':
                best_C = "{0:.6}".format(1.0*best_pos[0])
                best_gamma = "{0:.6}".format(best_pos[1])
            elif optimizer == 'GridSearchSVM':
                best_C = f'{best_pos[0]} (2^{int(np.log2(best_pos[0]))})'
                best_gamma = f'{best_pos[1]} (2^{int(np.log2(best_pos[1]))})'
            model_solution = pd.DataFrame(
                [best_C, best_gamma, "{0:.2%}".format(best_fit)],
                index=['Best C', 'Best Gamma',
                       'Best Fitness (Avg. Training Acc.)'],
                columns=['Value']
            )
            st.table(model_solution)

        st.header('Grasshoppers Movement')
        if optimizer == 'GOASVM':
            fig = px.scatter_3d(
                movement,
                x='C',
                y='gamma',
                z='fitness',
                color='generation',
                width=700,
                height=700
            )
        elif optimizer == 'GridSearchSVM':
            fig = px.scatter_3d(
                movement,
                x='C',
                y='gamma',
                z='fitness',
                width=700,
                height=700
            )
        st.write(fig)
        # st.dataframe(movement)

        st.header(f'Training Samples ({used_dataset})')

        if optimizer == 'GOASVM':
            train_sample = model.samples
            train_sample['target'] = model.targets
            test_sample = model.test_samples
        elif optimizer == 'GridSearchSVM':
            train_sample = model.X_train
            train_sample['target'] = model.y_train
            test_sample = model.X_test
            test_sample['target'] = model.y_test
            test_sample['prediction'] = model.y_pred

        st.write(train_sample.shape[0], 'samples')
        st.dataframe(train_sample)
        # st.write('Table above showing 1% of data.')

        st.header(f'Test Samples + Prediction ({used_dataset})')
        st.write(test_sample.shape[0], 'samples')
        st.dataframe(test_sample)
        # st.write('Table above showing 5% of data.')

        st.header('Confusion Matrix')
        fig = plt.figure()
        conf_matrix = confusion_matrix(
            test_sample['target'], test_sample['prediction'])
        sns.heatmap(
            conf_matrix,
            cmap=sns.color_palette("light:b", as_cmap=True),
            cbar=False,
            annot=True,
            xticklabels=np.unique(test_sample['target']),
            yticklabels=np.unique(test_sample['target']),
            fmt="d"
        )
        plt.ylabel("Actual", fontweight='bold')
        plt.xlabel("Predicted", fontweight='bold')
        st.pyplot(fig)
        accuracy = accuracy_score(
            test_sample['target'], test_sample['prediction']) * 100
        st.write('**Accuracy (%): **: ', accuracy)
    else:
        st.markdown(
            '<span style="color:red">No models have been saved yet.</span>', True)
