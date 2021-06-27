import streamlit as st
# import plotly.express as px
import numpy as np
import pandas as pd


def app():
    gpt = pd.read_csv('data/gpt.csv')
    gpt['file_path'] = gpt['file_path'].str[1:]

    gpt_split = pd.read_csv('data/gpt_split.csv')
    gpt_split['file_path'] = gpt_split['file_path'].str[1:]

    st.title('Dataset')
    st.write('<hr>', unsafe_allow_html=True)

    st.write(
        'The Guitar Playing Technique (GPT) datasets from the work of [Su et al. (2014)](http://mac.citi.sinica.edu.tw/GuitarTranscription/) was utilized.')
    st.write('This dataset comprises `7 playing techniques` of the electrical guitar, including: `bending`, `hamming`, `mute`, `normal`, `pulling`, `slide`, and `trill`')
    st.write('There are two sets of data:')
    st.write('1. A `complete dataset`, which includes complete audio signals of each guitar sound with a duration of `4.0 s`.')
    st.write('2. A `split dataset`, which includes only portions of the waveform signals on the onsets of each guitar sound, obtained by clipping them from `0.1 s` before the onset to `0.2 s` after the onset.')
    st.write('To make the quality of the sound recordings akin to that of real-world performance, `7 different guitar tones` are used with differences in effect and equalizer settings.')
    st.markdown('<font size="2"><table> \
                    <tr> \
                        <th style="width:20%">Tone name</th> \
                        <th>Effect</th> \
                        <th>Equalizer</th> \
                    </tr> \
                    <tr> \
                        <td>1 (Normal tone)</td> \
                        <td>moderate distortion</td> \
                        <td>no modification on EQ</td> \
                    </tr> \
                    <tr> \
                        <td>2 (Solo tone)</td> \
                        <td>moderate distortion and moderate reverb</td> \
                        <td>mid-frequency is emphasized</td> \
                    </tr> \
                    <tr> \
                        <td>3 (Solo tone)</td> \
                        <td>moderate distortion, intense chorus, slight reverb</td> \
                        <td>mid-frequency is emphasized</td> \
                    </tr> \
                    <tr> \
                        <td>4 (Solo tone)</td> \
                        <td>moderate distortion, intense delay, moderate reverb</td> \
                        <td>mid-frequency is emphasized</td> \
                    </tr> \
                    <tr> \
                        <td>5 (Riff tone)</td> \
                        <td>intense distortion</td> \
                        <td>mid-frequency is suppressed while high-frequency and low-frequency are emphasized</td> \
                    </tr> \
                    <tr> \
                        <td>6 (Country tone)</td> \
                        <td>very slight distortion</td> \
                        <td>no modification on EQ</td> \
                    </tr> \
                    <tr> \
                        <td>7 (Funk tone)</td> \
                        <td>slight distortion, slight delay, and slight reverb</td> \
                        <td>high-frequency component is emphasized a little</td> \
                    </tr> \
                </table></font>', unsafe_allow_html=True)
    st.write('<hr>', unsafe_allow_html=True)

    st.header('1. GPT-complete Dataset')
    st.subheader("Number of Sound Clips in GPT-Complete Dataset")
    st.write('*Total:', gpt.shape[0], ' audio files.*')
    st.bar_chart(pd.value_counts(gpt['technique']))

    st.subheader('Play an Audio Clip of GPT-complete Dataset')
    techniques = gpt['technique'].unique()
    tones = gpt['tone_type'].unique()
    selected_technique = st.selectbox('Select Technique:', np.sort(techniques))
    selected_tone = st.selectbox('Select Tone Type:', np.sort(tones))
    files = gpt['file_path'].loc[(gpt['technique'] == selected_technique) & (
        gpt['tone_type'] == selected_tone)].sort_values()
    df_files = files.to_frame()
    df_files['value'] = np.array(files.str.split('/').tolist())[:, 5]
    selected_file = st.selectbox('Select File:', df_files['value'].tolist())
    selected_file_path = df_files['file_path'].loc[df_files['value']
                                                   == selected_file].item()
    st.write('`Play: ', selected_file_path, '`')
    audio_file = open(selected_file_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes)
    st.write('<hr>', unsafe_allow_html=True)

    st.header('2. GPT-split Dataset')
    st.subheader("Number of Sound Clips in GPT-split Dataset")
    st.write('*Total:', gpt_split.shape[0], ' audio files.*')
    st.bar_chart(pd.value_counts(gpt_split['technique']))

    st.subheader('Play an Audio Clip of GPT-split Dataset')
    techniques2 = gpt_split['technique'].unique()
    tones2 = gpt_split['tone_type'].unique()
    selected_technique2 = st.selectbox(
        'Select Technique', np.sort(techniques2))
    selected_tone2 = st.selectbox('Select Tone Type', np.sort(tones2))
    files2 = gpt_split['file_path'].loc[(gpt_split['technique'] == selected_technique2) & (
        gpt_split['tone_type'] == selected_tone2)].sort_values()
    df_files2 = files2.to_frame()
    df_files2['value'] = np.array(files2.str.split('/').tolist())[:, 5]
    selected_file2 = st.selectbox('Select File', df_files2['value'].tolist())
    selected_file_path2 = df_files2['file_path'].loc[df_files2['value']
                                                     == selected_file2].item()
    st.write('`Play: ', selected_file_path2, '`')
    audio_file2 = open(selected_file_path2, 'rb')
    audio_bytes2 = audio_file2.read()
    st.audio(audio_bytes2)
    st.write('<hr>', unsafe_allow_html=True)

    st.header('Extracted Features of GPT Datasets')
    st.write('To represent musical signal, the `mean`, `std`, `max`, `median`, `min`, `skewness`, and `kurtosis` as the statistics measure of various audio descriptors including: *MFCC-13*, *$\Delta$MFCC-13* (first-order derivative), *$\Delta$<sub>2</sub>MFCC-13* (second-order derivative) was utilized.', unsafe_allow_html=True)
    st.latex(r'''
        Total = 7 \times 13 \times 3 = 273D \ Feature \ Vector
        ''')
    # st.write('The audio descriptors are computed using python package for music and audio analysis, [librosa](https://librosa.org/doc/latest/index.html).')
    st.markdown('### GPT-complete dataset (1% sampling)')
    sample_gpt = gpt.sample(frac=0.01, random_state=0)
    st.dataframe(sample_gpt)

    st.markdown('### GPT-split dataset (1% sampling)')
    sample_gpt_split = gpt_split.sample(frac=0.01, random_state=0)
    st.dataframe(sample_gpt_split)
