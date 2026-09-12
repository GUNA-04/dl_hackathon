"""JANUS Streamlit demo: prevention, audit, and experiment evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="JANUS — Controllable Forgetting", layout="wide")
st.title("JANUS: Bidirectional Control of Catastrophic Forgetting")
st.caption("Predict interference → protect vulnerable classes → audit a deletion request.")
df = pd.read_csv("results/results.csv")
risk_path = Path("results/risk_scores.csv")
risk = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame()

summary = df.groupby("method")[["final_accuracy", "avg_forgetting"]].agg(["mean", "std"])
st.subheader("Evidence from the reproducible benchmark")
st.dataframe(summary.style.format("{:.3f}"), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("PREVENT — fixed-budget memory allocation")
    budget = st.selectbox("Memory budget", sorted(df.memory_size.unique()), index=1)
    view = df[df.memory_size == budget].groupby("method", as_index=False).avg_forgetting.mean()
    fig, ax = plt.subplots()
    ax.bar(view.method, view.avg_forgetting, color=["#d55e00" if x == "iar" else "#6c757d" for x in view.method])
    ax.set_ylabel("Average forgetting ↓"); ax.set_title(f"Forgetting at N={budget}")
    st.pyplot(fig); plt.close(fig)
with right:
    st.subheader("INDUCE — deletion-request audit")
    st.warning("Targeted unlearning is an experimental diagnostic, not certified machine unlearning.")
    target = st.number_input("Requested class ID", 0, 99, 0)
    st.write(f"Deletion request for class **{target}**: remove its replay exemplars, then replay its closest interferers.")
    st.metric("Proof gate", "Pending dedicated unlearning run")

st.subheader("Why is a class at risk?")
if risk.empty:
    st.info("Run the IAR experiment to populate per-class risk scores.")
else:
    row = risk.sort_values("predicted_risk", ascending=False).iloc[0]
    st.write(f"Class **{int(row.class_id)}** is a high-priority replay candidate: predicted interference risk is **{row.predicted_risk:.3f}** on transition {int(row.transition)}.")

st.subheader("Accuracy matrices")
files = sorted(Path("results/matrices").glob("*.npy"))
if files:
    selected = st.selectbox("Matrix", files, format_func=lambda p: p.stem)
    fig, ax = plt.subplots(); im = ax.imshow(np.load(selected), cmap="viridis")
    ax.set(xlabel="Evaluated task", ylabel="Training through task", title=selected.stem); fig.colorbar(im, ax=ax)
    st.pyplot(fig); plt.close(fig)
st.caption("Single source of truth: results/results.csv. Validation data is used during continual training; any final test claim must be reported separately.")
