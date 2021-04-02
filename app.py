import streamlit as st
from multiapp import MultiApp
from apps import home, eda # import app modules here

app = MultiApp()

# Add all application here
app.add_app("Home", home.app)
app.add_app("EDA", eda.app)

# The main app
app.run()