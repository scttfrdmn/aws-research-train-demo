"""Streamlit collaborator viewer (SPEC §4 stage 7, §5).

The frozen, drop-in-your-input surface: a scientist supplies their own input
(a SMILES string, later a weather patch / sequence / prompt) and sees the
trained model's prediction and the base-vs-trained behavior delta. The proof was
banked upstream in the sweep; this surface proves taste, not skill.

Domain-blind: it loads the head by name and dispatches to ``head.viewer()`` —
it never branches on which domain is loaded (mirrors the spine, SPEC §5).

Serve behind the Studio proxy (verify-first #3):

    streamlit run app/app.py --server.port 8501 \
      --server.baseUrlPath "jupyterlab/default/proxy/8501" \
      --server.headless true --server.enableXsrfProtection false \
      --server.enableCORS false
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from spine import registry  # noqa: E402

st.set_page_config(page_title="Research Train — viewer", layout="wide")
st.title("Drop in your input")
st.caption("the endpoints prove taste, not skill — the proof was banked upstream")

domains = registry.names()
if not domains:
    st.warning("No heads registered.")
    st.stop()

with st.sidebar:
    domain = st.selectbox("domain", domains)
    checkpoint = st.text_input("checkpoint path", "checkpoints/model.pt")

head = registry.load(domain)
viewer = head.viewer()

# Domain-native default input. The viewer owns rendering; the app just collects
# one input and hands it to the head — no per-domain branching here.
default_input = {"molecular": "CCO"}.get(domain, "")
user_input = st.text_input(f"{domain} input", default_input)

col_base, col_tuned = st.columns(2)
if st.button("predict") and user_input:
    if not Path(checkpoint).exists():
        st.error(f"checkpoint not found: {checkpoint}")
    else:
        with col_tuned:
            st.subheader("trained")
            rendered = viewer.render(checkpoint, user_input)
            st.json(rendered)
        with col_base:
            st.subheader("input")
            st.write(user_input)
