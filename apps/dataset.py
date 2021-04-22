import streamlit as st
# import plotly.express as px
import numpy as np
import pandas as pd


def app():
    gpt = pd.read_csv('data/gpt.csv')
    gpt['file_path'] = gpt['file_path'].str[1:]

    st.title('Dataset')
    st.write(
        'The Guitar Playing Technique (GPT) dataset from the work of [Su et al. (2014)](http://mac.citi.sinica.edu.tw/GuitarTranscription/) was utilized.')

    st.header("Number of Sound Clips in GPT Dataset")
    st.write('This dataset comprises 7 playing techniques of the electrical guitar with 7 different guitar tones. ***(Total:',
             gpt.shape[0], ' samples)***.')
    st.bar_chart(pd.value_counts(gpt['technique']))

    st.header('Play Audio Clip of GPT')
    techniques = gpt['technique'].unique()
    tones = gpt['tone_type'].unique()
    selected_technique = st.selectbox('Select Technique', np.sort(techniques))
    selected_tone = st.selectbox('Select Tone Type', np.sort(tones))
    files = gpt['file_path'].loc[(gpt['technique'] == selected_technique) & (
        gpt['tone_type'] == selected_tone)].sort_values()
    df_files = files.to_frame()
    df_files['value'] = np.array(files.str.split('/').tolist())[:, 5]
    selected_file = st.selectbox('Select File', df_files['value'].tolist())
    selected_file_path = df_files['file_path'].loc[df_files['value']
                                                   == selected_file].item()
    st.write('Play: ', selected_file_path)
    audio_file = open(selected_file_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)

    st.header('GPT Dataset Samples')
    st.write('To represent musical signal, the **mean, standard deviation, maximum, median, minimum, skewness, and kurtosis** as the statistics measure of various audio descriptors such as: **MFCC**, **$\Delta$MFCC** (first-order derivative), **$\Delta\Delta$MFCC** (second-order derivative) was utilized.')
    st.write(
        'The audio descriptors are computed using python package for music and audio analysis, [librosa](https://librosa.org/doc/latest/index.html).')
    st.write('The table below shows 1% of the sample data from the dataset.')
    sample_gpt = gpt.sample(frac=0.01, random_state=0)
    st.dataframe(sample_gpt)
