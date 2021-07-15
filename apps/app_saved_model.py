import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import streamlit as st
from scipy.spatial import Delaunay
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef)

from .constants import RANDOM_STATE


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
            sigma_range = (model.lb[1], model.ub[1])
            if optimizer == 'GOASVM':
                model_parameter = pd.DataFrame(
                    [k_fold, pop_size, epoch, c_minmax, c_range, sigma_range],
                    index=['K-Fold', 'Pop. Size', 'Maximum Iteration',
                           'c_minmax', 'log\u2082C Range', 'log\u2082\u03c3 Range'],
                    columns=['Value']
                )
            elif optimizer == 'GridSearchSVM':
                model_parameter = pd.DataFrame(
                    [k_fold, f'[{c_range[0]},...,{c_range[1]}]',
                        f'[{sigma_range[0]},...,{sigma_range[1]}]', model.step_size],
                    index=['K-Fold', 'log\u2082C Set',
                           'log\u2082\u03c3 Set', 'Step'],
                    columns=['Value']
                )
            st.table(model_parameter)
        with col2:
            st.header('Solution')
            best_pos = model.solution[0]
            best_fit = model.solution[1]
            if optimizer == 'GOASVM':
                best_C = f'{2**best_pos[0]:.4f} ({best_pos[0]:.4f})'
                best_sigma = f'{2**best_pos[1]:.4f} ({best_pos[1]:.4f})'
            elif optimizer == 'GridSearchSVM':
                best_C = f'{best_pos[0]:.4f} ({np.log2(best_pos[0])})'
                best_sigma = f'{best_pos[1]:.4f} ({np.log2(best_pos[1])})'
            indeks = ['Best C (log\u2082C)',
                      'Best \u03c3 (log\u2082\u03c3)', 'Fitness*']
            model_solution = pd.DataFrame(
                [best_C, best_sigma, "{0:.4f}".format(best_fit)],
                index=indeks,
                columns=['Value']
            )
            st.table(model_solution)

        if optimizer == 'GOASVM':
            movement['C'] = np.round(movement['C'], decimals=4)
            movement['Sigma'] = np.round(movement['Sigma'], decimals=4)
            movement['Fitness'] = np.round(movement['Fitness'], decimals=4)
            st.header('Grasshoppers Movement')
            fig = px.scatter_3d(
                movement,
                x='C',
                y='Sigma',
                z='Fitness',
                color='Iteration',
                labels={
                    "C": "log<sub>2</sub>C",
                    "Sigma": "log<sub>2</sub>\u03C3"
                }
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=c_range,),
                    yaxis=dict(range=sigma_range,),
                )
            )
        elif optimizer == 'GridSearchSVM':
            st.header('Validation Accuracy')
            df = movement
            movement = movement.pivot(
                index='C', columns='Sigma', values='Fitness')
            fig = px.imshow(
                np.asmatrix(movement),
                labels=dict(x="log<sub>2</sub>\u03C3",
                            y="log<sub>2</sub>C", color="Fitness"),
                x=[str(x) for x in movement.columns],
                y=[str(x) for x in movement.index],
                width=600,
                height=600
            )
            fig.update_xaxes(side="top")

            x = df['C']
            y = df['Sigma']
            z = df['Fitness']
            points2D = np.vstack([x, y]).T
            tri = Delaunay(points2D)
            simplices = tri.simplices

            trisurf = ff.create_trisurf(
                x=x, y=y, z=z,
                colormap=(px.colors.sequential.Plasma),
                simplices=simplices,
                title=None,
                width=600,
                height=600
            )
            st.write(trisurf)
        st.write(fig)

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

        n_train = np.ceil(
            train_sample.shape[0]/(train_sample.shape[0]+test_sample.shape[0])*100)
        n_test = 100 - n_train

        st.header(f'Train Samples ({used_dataset})')
        st.write(
            f'`{train_sample.shape[0]}` samples | `{n_train:.0f}%` of total data. ')
        st.dataframe(train_sample.sample(frac=frac, random_state=RANDOM_STATE))
        st.text(f'Table above showing only {int(frac*100)}% of train samples.')

        st.header(f'Test Samples + Prediction ({used_dataset})')
        st.write(
            f'`{test_sample.shape[0]}` samples | `{n_test:.0f}%` of total data')
        st.dataframe(test_sample.sample(frac=frac, random_state=RANDOM_STATE))
        st.text(f'Table above showing only {int(frac*100)}% of test samples.')

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

        clf_report = classification_report(
            test_sample['target'], test_sample['prediction'])
        print(clf_report)
        mcc = matthews_corrcoef(
            test_sample['target'], test_sample['prediction'])
        st.subheader(f'MCC: `{mcc:.4f}`')
    else:
        st.markdown(
            '<span style="color:red">No model has been saved yet.</span>', True)
