"""Streamlit front end for the ExtraaLearn lead-conversion model."""
import os
import pandas as pd
import requests
import streamlit as st

# URL of the Flask backend Space, e.g. https://<user>-<space>.hf.space
API_URL = os.getenv("API_URL", "https://YOUR-USERNAME-extraalearn-backend.hf.space")

st.set_page_config(page_title="ExtraaLearn Lead Scorer", page_icon="🎓", layout="centered")
st.title("🎓 ExtraaLearn – Lead Conversion Predictor")
st.write("Estimate how likely a lead is to become a paying customer.")

tab1, tab2 = st.tabs(["Single lead", "Batch (CSV)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 70, 45)
        occupation = st.selectbox("Current occupation", ["Professional", "Unemployed", "Student"])
        first = st.selectbox("First interaction", ["Website", "Mobile App"])
        profile = st.selectbox("Profile completed", ["High", "Medium", "Low"])
        last = st.selectbox("Last activity", ["Email Activity", "Phone Activity", "Website Activity"])
        visits = st.number_input("Website visits", 0, 50, 3)
    with c2:
        time_spent = st.number_input("Time spent on website (seconds)", 0, 5000, 600)
        pages = st.number_input("Page views per visit", 0.0, 30.0, 3.0, step=0.1)
        p1 = st.selectbox("Saw print media type 1", ["No", "Yes"])
        p2 = st.selectbox("Saw print media type 2", ["No", "Yes"])
        dig = st.selectbox("Saw digital media", ["No", "Yes"])
        edu = st.selectbox("Educational channels", ["No", "Yes"])
        ref = st.selectbox("Referral", ["No", "Yes"])

    if st.button("Predict", type="primary"):
        lead = {"age": age, "current_occupation": occupation, "first_interaction": first,
                "profile_completed": profile, "website_visits": visits,
                "time_spent_on_website": time_spent, "page_views_per_visit": pages,
                "last_activity": last, "print_media_type1": p1, "print_media_type2": p2,
                "digital_media": dig, "educational_channels": edu, "referral": ref}
        try:
            r = requests.post(f"{API_URL}/predict", json=lead, timeout=30)
            r.raise_for_status()
            out = r.json()
            prob = out["conversion_probability"]
            st.metric("Conversion probability", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            (st.success if prob >= 0.5 else st.warning)(out["prediction"])
        except Exception as e:
            st.error(f"Could not reach the prediction API: {e}")

with tab2:
    st.write("Upload a CSV with the same columns as the training data (an `ID` column is optional).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None and st.button("Score leads"):
        try:
            r = requests.post(f"{API_URL}/predict_batch", files={"file": up.getvalue()}, timeout=120)
            r.raise_for_status()
            res = pd.DataFrame(r.json()["results"]).sort_values("conversion_probability", ascending=False)
            st.dataframe(res, use_container_width=True)
            st.download_button("Download scores", res.to_csv(index=False), "scored_leads.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not score the file: {e}")
