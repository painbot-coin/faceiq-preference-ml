"""Standalone entry for the pilot-500 labeler (also available as a dashboard tab).

Run:  streamlit run app/labeler.py
"""

from __future__ import annotations

import streamlit as st

from labeler_ui import render_labeler

st.set_page_config(page_title="Pilot-500 Labeler", layout="wide")
st.title("Pilot-500 Labeler")
render_labeler()
