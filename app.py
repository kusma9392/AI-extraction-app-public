import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
# Load dataset
data = pd.read_csv("data.csv")

X = data[['Solvent', 'Time', 'Temp']]
y = data['Yield']

# Train model
model = DecisionTreeRegressor()
model.fit(X, y)

# UI
st.title("🧪 AI Extraction Yield Predictor")

# Input fields
solvent_dict = {
    "Methanol": 1,
    "Water": 2,
    "Benzene": 3,
    "Hexane": 4
}

solvent_name = st.selectbox("Select Solvent", list(solvent_dict.keys()))
solvent = solvent_dict[solvent_name]
time = st.slider("Extraction Time", 10, 100)
temp = st.slider("Temperature", 20, 100)

# Prediction
if st.button("Predict"):
    result = model.predict([[solvent, time, temp]])
    st.success(f"Predicted Yield ≈ {result[0]:.2f}%")

    st.subheader("📊 Yield Comparison (Bar Graph)")

# Group data
avg_data = data.groupby('Solvent')['Yield'].mean()

# Create figure
fig, ax = plt.subplots()

# Plot bar graph
ax.bar(avg_data.index, avg_data.values)

# Labels
ax.set_xlabel("Solvent")
ax.set_ylabel("Average Yield")

# Show graph in Streamlit
st.pyplot(fig)   


