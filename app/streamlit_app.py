import streamlit as st


analysis_page = st.Page(
  page='pages/analysis.py',
  title='Data Exploration and Analysis',
  icon=':material/account_circle:',
  default=True
)


deployed_model = st.Page(
  page='pages/deployed_model.py',
  title='Deployed Model',
  icon=':material/account_circle:',
  default=False
)

evaluation_page = st.Page(
  page='pages/evaluation.py',
  title='Model Evaluation',
  icon=':material/account_circle:',
  default=False
)

