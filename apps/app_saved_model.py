import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def app():
    path = './data/misc/'
    st.title('Load Saved Models')

    list_file = os.listdir(path)
    list_file = [filename[:-4] for filename in list_file]
    list_file.sort()
    filename = st.selectbox('Select Model:', list(dict.fromkeys(list_file)))

    if filename:
        used_dataset = filename.split('_')[0]
        optimizer = filename.split('_')[1]
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
                    [k_fold, f'[{c_range[0]},...,{c_range[1]}]',
                        f'[{gamma_range[0]},...,{gamma_range[1]}]'],
                    index=['K-Fold', 'C Set (Exp. of 2)',
                           'Gamma Set (Exp. of 2)'],
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
                indeks = ['Best C', 'Best Gamma', 'Best Fitness']
            elif optimizer == 'GridSearchSVM':
                best_C = f'{best_pos[0]} (2^{int(np.log2(best_pos[0]))})'
                best_gamma = f'{best_pos[1]} (2^{int(np.log2(best_pos[1]))})'
                indeks = ['Best C', 'Best Gamma', 'Val. Accuracy']
            model_solution = pd.DataFrame(
                [best_C, best_gamma, "{0:.2%}".format(best_fit)],
                index=indeks,
                columns=['Value']
            )
            st.table(model_solution)

        if optimizer == 'GOASVM':
            st.header('Grasshoppers Movement')
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
            st.header('Validation Accuracy')
            movement = movement.pivot(
                index='C', columns='gamma', values='fitness')
            fig = px.imshow(
                np.asmatrix(movement),
                labels=dict(x="Gamma", y="C", color="Val. Acc."),
                x=[str(x) for x in movement.columns],
                y=[str(x) for x in movement.index],
                width=700,
                height=700,
                color_continuous_scale='RdBu_r'
            )
            fig.update_xaxes(side="top")
        st.write(fig)
        # st.dataframe(movement)

        st.header(f'Train Samples ({used_dataset})')

        frac = 0.01
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

        st.write(train_sample.shape[0], 'samples | ', int(np.ceil(
            train_sample.shape[0]/(train_sample.shape[0]+test_sample.shape[0])*100)), '% of total data.')
        st.dataframe(train_sample.sample(frac=frac, random_state=0))
        st.write(f'Table above showing {int(frac*100)}% of train samples.')

        st.header(f'Test Samples + Prediction ({used_dataset})')
        st.write(test_sample.shape[0], 'samples | ', int(np.floor(
            test_sample.shape[0]/(train_sample.shape[0]+test_sample.shape[0])*100)), '% of total data.')
        st.dataframe(test_sample.sample(frac=frac, random_state=0))
        st.write(f'Table above showing {int(frac*100)}% of test samples.')

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
        st.write('**Test Accuracy (%): **: ', accuracy)
        # st.write(classification_report(test_sample['target'], test_sample['prediction']))
    else:
        st.markdown(
            '<span style="color:red">No models have been saved yet.</span>', True)
