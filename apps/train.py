import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from methods.goa_svm import GOA_SVM
import pickle
from sklearn.metrics import accuracy_score


def app():
    st.title('Training & Testing')

    st.header('Dataset')
    col1, col2 = st.beta_columns(2)
    with col1:
        selected_dataset = st.selectbox('Select Dataset:', ['GPT', 'Iris'])
    with col2:
        train_size = st.number_input('Train Size:', 0.1, 0.9, 0.80)

    if selected_dataset == 'Iris':
        df = pd.read_csv('data/iris.csv')
        X = df.iloc[:, :4]
        y = df['variety']
    elif selected_dataset == 'GPT':
        df = pd.read_csv('data/gpt.csv')
        X = df.iloc[:, :273]
        y = df['technique']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42)
    dataset_summarize = pd.DataFrame(
        [[X_train.shape[1], X_train.shape[0], X_test.shape[0], X.shape[0]]],
        columns=['Num. Features', 'Num. Train Samples',
                 'Num. Test Samples', 'Total']
    )
    st.table(dataset_summarize.assign(hack='').set_index('hack'))

    st.header('Parameter Setting')
    k_fold = st.number_input('K-Fold :', 1, 10, 3, step=1)
    col1, col2 = st.beta_columns(2)
    with col1:
        range_C = st.slider(
            "Parameter C (regularization) range:", 1.0, 1000.0, (1.0, 500.0), 1.0)
        range_gamma = st.slider("Parameter gamma range: ",
                                0.01, 0.5, (1e-02, 0.1), 1e-02)
        c_min = st.number_input('c_min :', 1e-05, 1.0,
                                1e-05, step=1e-05, format="%.5f")
        verbose = st.checkbox('Show backend output.', value=True)
        save = st.checkbox('Save the model for analysis.')
        if save:
            filename = st.text_input(
                'Filename:', help='Use _ for word divider.')

    with col2:
        pop_size = st.number_input(
            'Population Size (number of grasshopper) :', 1, 100, 30, step=1)
        epoch = st.number_input('Epoch :', 1, 100, 10, step=1)
        c_max = st.number_input('c_max :', 1)

    if st.button('Train & Test'):
        lb = [range_C[0], range_gamma[0]]
        ub = [range_C[1], range_gamma[1]]
        c_minmax = (c_min, c_max)
        # print(k_fold, lb, ub, pop_size, epoch, c_minmax)

        st.write('**Train using ', '{:.1%}'.format(train_size),
                 'of dataset samples.**', X_train.shape)

        md = GOA_SVM(k_fold=k_fold, lb=lb, ub=ub, verbose=verbose,
                     pop_size=pop_size, epoch=epoch, c_minmax=c_minmax)
        best_pos, best_fit, _ = md.train(X_train, y_train)
        st.write('The training process has been completed.')
        st.write('Best Parameter (C, gamma): ', (best_pos[0], best_pos[1]))
        st.write('Best Training Avg. Accuracy: ', best_fit)

        st.write('**Test using ', '{:.1%}'.format(1.0-train_size),
                 'of dataset samples.**', X_test.shape)

        y_pred = md.predict(X_test, y_test)
        st.write('Final Accuracy Test: ', accuracy_score(y_pred, y_test))

        if save:
            md.save_movement_to_csv(filename)
            name = './data/misc/'+filename+'.sav'
            pickle.dump(md, open(name, 'wb'))
            st.markdown(
                '<span style="color:blue">*The model has been saved.*</span>', True)
