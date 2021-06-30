import streamlit as st
import pandas as pd
import numpy as np
import scipy.fftpack
from scipy.stats import skew, kurtosis
import pickle
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import librosa
import librosa.display


def app():
    st.title('GPT Classification')
# 1
    st.header('Select Model')
    path = './data/misc/'
    list_file = os.listdir(path)
    list_file = [filename[:-4] for filename in list_file]
    list_file.sort()
    model_name = st.selectbox('Select Model:', list(dict.fromkeys(list_file)))
    
    if model_name:
        model = open(path+model_name+'.sav', 'rb')
        model = pickle.load(model)
        
        dataset = model_name.split('_')[0]
        if dataset == "GPT Split":
            data_path = "./data/gpt_split.csv"
        elif dataset == "GPT Complete":
            data_path = './data/gpt.csv'
        
        data = pd.read_csv(data_path)
        audio_path = data['file_path']
        y = data['technique']
        _, X_test, _, y_test = train_test_split(audio_path, y, train_size=0.9, random_state=0)
        data_test = pd.DataFrame(X_test)
        # data_test['technique'] = y_test
        data_test = data_test.sort_values('file_path')
        # st.dataframe(data_test)
# 2
        st.header('2. Select GPT Audio File')
        selected_audio = st.selectbox("Select GPT Audio:", list(data_test['file_path']))
        selected_audio_path = selected_audio[1:]
        technique = selected_audio.split('/')[4].split('_')[0]
        audio_file = open(selected_audio_path, 'rb')
        audio_samples = audio_file.read()
        st.audio(audio_samples)
        st.write(f'Technique: `{technique}`')
# 3
        st.header('Feature Extraction')
        signal, sr = librosa.load(selected_audio_path)
# # 3a        
#         st.subheader('a. Load Audio Signal')
#         wave, ax = plt.subplots(figsize=(10, 1))
#         librosa.display.waveplot(signal, sr, alpha=0.5)
#         ax.set_ylabel('Amplitude')
#         st.write(wave)
# # 3b
#         hop_length = 512
#         n_fft = 2048
#         power = 2.0
#         win_length = None
#         center = True
#         pad_mode = 'reflect'
#         window = 'hann'

#         st.subheader('b. Short-time Fourier Transform (STFT)')
#         S = (np.abs(
#                 librosa.stft(
#                     signal,
#                     n_fft=n_fft,
#                     hop_length=hop_length,
#                     win_length=win_length,
#                     center=center,
#                     window=window,
#                     pad_mode=pad_mode,
#                 )
#             )** power
#         )
#         spectrogram, ax1 = plt.subplots(figsize=(10,5))
#         img_s = librosa.display.specshow(S, x_axis='time', y_axis='log')
#         spectrogram.colorbar(img_s, ax=ax1, label='Power')
#         ax1.set_ylabel('Frequency')
#         st.write(spectrogram)
# # 3c    
#         st.subheader('c. Mel Frequency Warping (Mel Scaling)')
#         M = librosa.feature.melspectrogram(signal, sr)
#         melspectrogram, ax2 = plt.subplots(figsize=(10,5))
#         img_m = librosa.display.specshow(S, x_axis='time', y_axis='mel')
#         melspectrogram.colorbar(img_m, ax=ax2, label='Power')
#         ax2.set_ylabel('Mel-Frequency')
#         st.write(melspectrogram)
# # 3d
#         st.subheader('d. Power to dB')
#         M_dB = librosa.power_to_db(M)
#         melspectrogram_db, ax3 = plt.subplots(figsize=(10,5))
#         img_mdb = librosa.display.specshow(M_dB, x_axis='time', y_axis='mel')
#         melspectrogram_db.colorbar(img_mdb, ax=ax3, format='%+2.0f dB')
#         ax3.set_ylabel('Mel-Frequency')
#         st.write(melspectrogram_db)

# # 3e 
#         dct_type = 2
#         norm = 'ortho'
#         st.subheader('e. Discrete Cosine Transform (DCT)')
#         dct = scipy.fftpack.dct(M_dB, axis=0, type=dct_type, norm=norm)
#         mfc, ax4 = plt.subplots(figsize=(10,5))
#         img_mfc = librosa.display.specshow(dct, x_axis='time')
#         mfc.colorbar(img_mfc, ax=ax4)
#         st.write(mfc)
# MFCC
        st.subheader('MFCC-13')
        mfcc = librosa.feature.mfcc(signal, sr, n_mfcc=13)
        fig_mfcc, ax5 = plt.subplots(figsize=(10,3))
        img_mfcc = librosa.display.specshow(mfcc, x_axis='time', ax=ax5)
        fig_mfcc.colorbar(img_mfcc, ax=ax5)
        st.write(fig_mfcc)
# Delta MFCC
        st.subheader('$\Delta$MFCC-13')
        delta_mfcc = librosa.feature.delta(mfcc)
        fig_dmfcc, ax6 = plt.subplots(figsize=(10,3))
        img_dmfcc = librosa.display.specshow(delta_mfcc, x_axis='time', ax=ax6)
        fig_dmfcc.colorbar(img_dmfcc, ax=ax6)
        st.write(fig_dmfcc)
# Delta2 MFCC
        st.subheader('$\Delta$$\Delta$MFCC-13')
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        fig_d2mfcc, ax7 = plt.subplots(figsize=(10,3))
        img_d2mfcc = librosa.display.specshow(delta2_mfcc, x_axis='time', ax=ax7)
        fig_d2mfcc.colorbar(img_d2mfcc, ax=ax7)
        st.write(fig_d2mfcc)
# Stat Measure
        st.subheader('Stats Measuring (Feature Vector)')
        mfccs = np.hstack((
                        np.mean(mfcc, axis=1),
                        np.std(mfcc, axis=1),
                        np.max(mfcc, axis=1),
                        np.median(mfcc, axis=1),
                        np.min(mfcc, axis=1),
                        skew(mfcc, axis=1),
                        kurtosis(mfcc, axis=1)
                    ))
        delta_mfccs = np.hstack((
                        np.mean(delta_mfcc, axis=1),
                        np.std(delta_mfcc, axis=1),
                        np.max(delta_mfcc, axis=1),
                        np.median(delta_mfcc, axis=1),
                        np.min(delta_mfcc, axis=1),
                        skew(delta_mfcc, axis=1),
                        kurtosis(delta_mfcc, axis=1)
                    ))
        delta2_mfccs = np.hstack((
                        np.mean(delta2_mfcc, axis=1),
                        np.std(delta2_mfcc, axis=1),
                        np.max(delta2_mfcc, axis=1),
                        np.median(delta2_mfcc, axis=1),
                        np.min(delta2_mfcc, axis=1),
                        skew(delta2_mfcc, axis=1),
                        kurtosis(delta2_mfcc, axis=1)
                    ))

        extracted_features = np.hstack((mfccs, delta_mfccs, delta2_mfccs))
        columns = []
        names = ['mfcc', 'delta_mfcc', 'delta2_mfcc']
        stats = ['mean', 'std', 'max', 'median', 'min', 'skew', 'kurtosis']

        for name in names:
            for stat in stats:
                for i in range(13):
                    col = f'{stat}_{name}_{i+1}'
                    columns = np.append(columns, col)

        df = pd.DataFrame([extracted_features], columns=columns)
        st.dataframe(df)
# Scaling
        st.subheader('Feature Normalization')
        df_scaled = model.min_max_scaler.transform(df)
        st.dataframe(df_scaled)

        if st.button('Classify / Predict'):
            prediction = model.model.predict(df_scaled)
            st.subheader(f'Result: `{prediction[0]}`')
    else:
        st.markdown(
            '<span style="color:red">No model has been saved yet.</span>', True)