from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COMPARE = RESULTS / "six_model_comparison" / "six_model_comparison.csv"

st.set_page_config(page_title="Ames Housing Model Dashboard", page_icon="🏠", layout="wide")
st.title("🏠 Ames Housing Price Prediction")
st.caption("AI/ML Group Project | Six individual regression models evaluated on a held-out 20% split of train.csv")

if not COMPARE.exists():
    st.error("Comparison results are missing. Run the six individual notebooks and the group comparison notebook first.")
    st.stop()

df = pd.read_csv(COMPARE).sort_values("holdout_rmsle").reset_index(drop=True)
best = df.iloc[0]

a, b, c, d = st.columns(4)
a.metric("Top model", best["model"])
b.metric("Best hold-out RMSLE", f"{best['holdout_rmsle']:.4f}")
c.metric("Typical error (MAE)", f"${best['mae']:,.0f}")
d.metric("R²", f"{best['r2']:.3f}")

st.info(f"**Current top result: {best['model']} ({best['member_id']}).** RMSLE is the error on the log price (lower is better); MAE and RMSE are in dollars.")

left, right = st.columns(2)
with left:
    st.subheader("Hold-out RMSLE by model (lower is better)")
    chart = df.set_index("model")[["holdout_rmsle"]].sort_values("holdout_rmsle", ascending=False)
    st.bar_chart(chart, x_label="RMSLE", y_label="Model", horizontal=True)
with right:
    st.subheader("Model metrics")
    view = df[["member_id", "model", "best_cv_rmsle", "holdout_rmsle", "rmse", "mae", "r2"]].copy()
    view.columns = ["Member ID", "Model", "CV RMSLE", "Hold-out RMSLE", "RMSE ($)", "MAE ($)", "R²"]
    st.dataframe(view.style.format({"CV RMSLE": "{:.4f}", "Hold-out RMSLE": "{:.4f}", "RMSE ($)": "{:,.0f}", "MAE ($)": "{:,.0f}", "R²": "{:.4f}"}),
                 use_container_width=True, hide_index=True)

st.subheader("Actual vs predicted price")
st.caption("Each dot is a hold-out house; the closer to the dashed diagonal, the better the prediction.")
tabs = st.tabs(df["model"].tolist())
for tab, (_, row) in zip(tabs, df.iterrows()):
    with tab:
        folder = RESULTS / "individual_models" / str(row["member_id"])
        st.write(f"**Member:** {row['member_id']}  |  **RMSLE:** {row['holdout_rmsle']:.4f}  |  **MAE:** ${row['mae']:,.0f}  |  **R²:** {row['r2']:.3f}")
        c1, c2 = st.columns(2)
        if (folder / "actual_vs_predicted.png").exists():
            c1.image(str(folder / "actual_vs_predicted.png"), caption=f"{row['model']} - hold-out", width=480)
        if (folder / "model_metrics.png").exists():
            c2.image(str(folder / "model_metrics.png"), caption="CV vs hold-out, errors, residuals")

with st.expander("How to read these scores"):
    st.markdown("**RMSLE:** error on log(price); measures relative error.  \n**RMSE / MAE:** typical error in dollars (RMSE punishes big misses more).  \n**R²:** share of price variation explained (1.0 = perfect).  \n\nThese are results on one 80/20 split of the Ames, Iowa data (2006-2010). They do not guarantee performance on other markets or years.")

st.caption("IT2011 AI/ML Group Project | Ames Housing | Results from the six member notebooks")
