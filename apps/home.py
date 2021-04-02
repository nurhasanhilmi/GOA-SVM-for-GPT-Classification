import streamlit as st
import pandas as pd
import numpy as np


def app():
    st.title('GOA-SVM for GPT Classification')

    st.write("This is an application for Classification of Guitar Playing Techniques (GPT) using the Grasshopper Optimization Algorithm (GOA) and Support Vector Machines (SVM).")

    st.markdown("### Sample Data")

    st.write('Navigate to `Data Stats` page to visualize the data')
