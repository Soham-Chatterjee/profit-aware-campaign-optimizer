"""Revenue AI Command Centre: entry point. Run with `uv run streamlit run app/Home.py`.

Each page renders the shared scenario sidebar itself (`common.sidebar()`); see D-038.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))   # lets pages import app/common.py

st.set_page_config(page_title="Profit-Aware Campaign Optimizer", page_icon="📈", layout="wide")

st.navigation([
    st.Page("views/0_Command_Centre.py", title="PACOPT Command Centre", default=True),
    st.Page("views/1_Customer_Intelligence.py", title="Customer Intelligence"),
    st.Page("views/2_Propensity_and_Opportunities.py", title="Propensity & Opportunities"),
    st.Page("views/3_Campaign_Optimization.py", title="Campaign Optimization"),
    st.Page("views/4_What_If_Simulation.py", title="What-if Simulation"),
]).run()
