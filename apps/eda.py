import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

def app():
    gpt = pd.read_csv('data/gpt.csv')
    gpt_split = pd.read_csv('data/gpt_split.csv')

    st.title('Exploratory Data Analysis')

    st.write("This is a simple exploratory data analysis from Guitar Playing Technique (GPT) dataset.")

    st.header('Dataset')

    st.write('The GPT dataset from the work of [Su et al. (2014)](http://mac.citi.sinica.edu.tw/GuitarTranscription/) was utilized.')
    st.write('This dataset comprises 7 playing techniques of the electrical guitar that is composed of 19 subclasses of GPT.')
    st.write('There are two sets of data:')

    st.subheader("1. GPT Complete Dataset")
    st.write('Includes complete audio signals of guitar sounds ***(total:', gpt.shape[0],' samples)***.')

    st.bar_chart(pd.value_counts(gpt['technique']))

    st.bar_chart(pd.value_counts(gpt['sub_technique']))

    st.subheader("2. GPT Split Dataset")
    st.write('Includes data on the onsets of sounds and only portions of the waveform signals, obtained by clipping them from 0.1 s before the onset to 0.2 s after the onset ***(total: ', gpt_split.shape[0], 'samples)***.')
    
    st.bar_chart(pd.value_counts(gpt_split['technique']))

    st.bar_chart(pd.value_counts(gpt_split['sub_technique']))