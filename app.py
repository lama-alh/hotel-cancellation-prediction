import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()

st.title("🏨 Hotel Booking Cancellation Predictor")
st.caption("Enter the booking details to estimate how likely the booking is to be canceled.")

col1, col2 = st.columns(2)

with col1:
    hotel = st.selectbox("Hotel type", ["City Hotel", "Resort Hotel"])
    lead_time = st.slider("Lead time (days before arrival)", 0, 700, 60)
    adr = st.number_input("Average daily rate (€)", min_value=0.0, max_value=1000.0, value=100.0, step=5.0)
    total_nights = st.slider("Total nights", 1, 30, 3)
    adults = st.slider("Adults", 1, 4, 2)
    booking_changes = st.slider("Booking changes", 0, 10, 0)

with col2:
    deposit_type = st.selectbox("Deposit type", ["No Deposit", "Non Refund", "Refundable"])
    market_segment = st.selectbox(
        "Booking channel",
        ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"],
    )
    customer_type = st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"])
    special_requests = st.slider("Special requests", 0, 5, 0)
    previous_cancellations = st.slider("Previous cancellations", 0, 10, 0)
    parking = st.slider("Car parking spaces", 0, 3, 0)

if st.button("Predict cancellation risk", type="primary", use_container_width=True):
    booking = pd.DataFrame([{
        "lead_time": lead_time,
        "adr": adr,
        "total_nights": total_nights,
        "adults": adults,
        "total_of_special_requests": special_requests,
        "previous_cancellations": previous_cancellations,
        "booking_changes": booking_changes,
        "required_car_parking_spaces": parking,
        "hotel": hotel,
        "deposit_type": deposit_type,
        "market_segment": market_segment,
        "customer_type": customer_type,
    }])

    prob = float(model.predict_proba(booking)[0, 1])

    st.divider()
    st.metric("Cancellation probability", f"{prob:.0%}")
    st.progress(prob)

    if prob >= 0.6:
        st.error("High risk: consider asking for a deposit or sending a confirmation reminder.")
    elif prob >= 0.3:
        st.warning("Medium risk: keep an eye on this booking.")
    else:
        st.success("Low risk: this booking is likely to be kept.")

st.divider()
st.caption("Model: Random Forest trained on the Hotel Booking Demand dataset (Kaggle).")
