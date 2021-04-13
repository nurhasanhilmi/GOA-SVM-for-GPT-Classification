import streamlit as st
from multiapp import MultiApp
from apps import home, dataset, train  # import app modules here

app = MultiApp()

# Add all application here
app.add_app("Home", home.app)
app.add_app("Dataset", dataset.app)
app.add_app("Training & Testing", train.app)

# The main app
app.run()