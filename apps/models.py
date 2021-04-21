import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def app():
    path = './data/misc/'
    st.title('Load Saved Models')

    list_file = os.listdir(path)
    list_file = [filename[:-4] for filename in list_file]
    filename = st.selectbox('Select Model:', list(dict.fromkeys(list_file)))

    if filename:
        movement = pd.read_csv(path+filename+'.csv')
        model = open(path+filename+'.sav', 'rb')
        model = pickle.load(model)

        col1, col2 = st.beta_columns(2)

        with col1:
            st.header('Model Parameter')
            epoch = model.epoch
            pop_size = model.pop_size
            c_minmax = model.c_minmax
            k_fold = model.k_fold
            c_range = (model.lb[0], model.ub[0])
            gamma_range = (model.lb[1], model.ub[1])
            model_parameter = pd.DataFrame(
                [k_fold, pop_size, epoch, c_minmax, c_range, gamma_range],
                index=['K-Fold', 'Pop. Size', 'Epoch',
                       'C_minmax (GOA)', 'C Range (SVM)', 'Gamma Range'],
                columns=['Value']
            )
            st.table(model_parameter)
        with col2:
            st.header('Model Solution')
            best_pos = model.solution[0]
            best_fit = model.solution[1]
            model_solution = pd.DataFrame(
                [best_pos[0], best_pos[1], "{0:.2%}".format(best_fit)],
                index=['Best Param C', 'Best Param Gamma',
                       'Best Fitness (Avg. Accuracy)'],
                columns=['Value']
            )
            st.table(model_solution)

        st.header('Grasshopper Movement Scatter Plot')
        fig = px.scatter_3d(
            movement,
            x='C',
            y='gamma',
            z='fitness',
            color='generation',
            width=700,
            height=700
        )
        st.write(fig)

        st.header('Training Samples')
        train_sample = model.samples
        train_sample['target'] = model.targets
        st.write(train_sample.shape[0], 'samples')
        st.dataframe(train_sample.sample(frac=0.01, random_state=0))
        st.write('Table above showing 1% of data.')

        st.header('Test Samples + Prediction')
        st.write(model.test_samples.shape[0], 'samples')
        st.dataframe(model.test_samples.sample(frac=0.01, random_state=0))
        st.write('Table above showing 1% of data.')

        st.header('Confusion Matrix')
        fig = plt.figure()
        conf_matrix = confusion_matrix(
            model.test_samples['target'], model.test_samples['prediction'])
        sns.heatmap(
            conf_matrix,
            cmap=sns.color_palette("light:b", as_cmap=True),
            annot=True,
            xticklabels=np.unique(model.test_samples['target']),
            yticklabels=np.unique(model.test_samples['target']),
            fmt="d"
        )
        plt.ylabel("True")
        plt.xlabel("Predicted")
        st.pyplot(fig)
        accuracy = accuracy_score(
            model.test_samples['target'], model.test_samples['prediction'])
        st.write('Accuracy: ', accuracy)
    else:
        st.markdown(
            '<span style="color:red">No models have been saved yet.</span>', True)
