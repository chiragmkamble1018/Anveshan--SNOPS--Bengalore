
import streamlit as st
import random
import time

st.title("🌍 Smart Disaster Alert System Dashboard")

temp = random.uniform(20, 50)
humidity = random.randint(30, 90)
gas = random.randint(100, 500)
vibration = random.choice(["Detected", "None"])

st.metric("Temperature (°C)", f"{temp:.2f}")
st.metric("Humidity (%)", f"{humidity}")
st.metric("Gas Level", gas)
st.metric("Vibration", vibration)

if gas > 400 or temp > 45:
    st.error("🚨 CRITICAL ALERT: Possible Fire or Gas Leak!")
elif humidity > 80:
    st.warning("⚠️ Warning: High Humidity Detected")
else:
    st.success("✅ System Normal")

time.sleep(1)