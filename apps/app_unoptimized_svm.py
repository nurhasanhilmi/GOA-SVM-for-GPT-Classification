import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from .constants import RANDOM_STATE, DATASET_PATH


def app():
    st.title('Unoptimized SVM')

    st.header('Dataset Setting')
    col1, col2 = st.beta_columns(2)
    with col1:
        sampling_size = st.number_input('Sampling Size (%):', 5, 100, 100, 5)
    with col2:
        train_size = st.number_input('Train Size (%):', 60, 90, 80, 5)

    df = pd.read_csv(DATASET_PATH)
    if sampling_size != 100:
        sampling_size = sampling_size/100
        df = df.sample(frac=sampling_size, random_state=RANDOM_STATE)
    X = df.iloc[:, :195]
    y = df['technique']
    y_sub = df['subtechnique']

    train_size = train_size/100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        random_state=RANDOM_STATE,
        stratify=y_sub
    )

    dataset_summarize = pd.DataFrame(
        [[X_train.shape[1], X_train.shape[0], X_test.shape[0], X.shape[0]]],
        columns=['Num. Features', 'Num. Train Samples',
                 'Num. Test Samples', 'Total Samples']
    )
    st.table(dataset_summarize.assign(hack='').set_index('hack'))

    if st.button('Train & Test'):
        st.write('**Start The Training Process**')

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X_train)
        X_train_ = scaler.transform(X_train)
        X_test_ = scaler.transform(X_test)
        clf = SVC(kernel='rbf', decision_function_shape='ovo')
        clf.fit(X_train_, y_train)

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