import streamlit as st
from multiapp import MultiApp
from apps import app_unoptimized_svm ,app_goa_svm, app_grid_search_svm, app_dataset, app_saved_model, app_gpt_classification # import app modules here

app = MultiApp()

# Add all application here
# app.add_app("GPT Classification", app_gpt_classification.app)
app.add_app("GOA-SVM", app_goa_svm.app)
app.add_app("Grid Search-SVM", app_grid_search_svm.app)
app.add_app("Unoptimized SVM", app_unoptimized_svm.app)
# app.add_app("Dataset", app_dataset.app)
app.add_app("Saved Models", app_saved_model.app)

# The main app
app.run()
