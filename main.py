import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLUMNS = ["temperature", "humidity", "light", "voltage"]

EPOCHS = 80
BATCH_SIZE = 64
LATENT_DIM = 2
LEARNING_RATE = 0.001


# =========================
# DATASET
# =========================

def create_iot_dataset(n=3000):
    temperature = np.random.normal(24, 3, n)
    humidity = np.random.normal(55, 10, n)
    light = np.random.normal(400, 120, n)
    voltage = np.random.normal(3.2, 0.2, n)

    return pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "light": light,
        "voltage": voltage
    })


# =========================
# VAE MODEL
# =========================

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def vae_loss(reconstructed, original, mu, logvar):
    reconstruction_loss = nn.functional.mse_loss(
        reconstructed,
        original,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return reconstruction_loss + kl_loss


# =========================
# TRAIN MODEL
# =========================

@st.cache_resource
def train_vae_model():
    real_df = create_iot_dataset()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(real_df)

    train_tensor = torch.tensor(scaled_data, dtype=torch.float32)

    model = VAE(
        input_dim=len(FEATURE_COLUMNS),
        latent_dim=LATENT_DIM
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(train_tensor.size(0))
        epoch_loss = 0

        for i in range(0, train_tensor.size(0), BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch = train_tensor[indices]

            optimizer.zero_grad()

            reconstructed, mu, logvar = model(batch)
            loss = vae_loss(reconstructed, batch, mu, logvar)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_tensor))

    return model, scaler, real_df, losses


# =========================
# GENERATION
# =========================

def generate_synthetic_data(model, scaler, count=1):
    model.eval()

    with torch.no_grad():
        z = torch.randn(count, LATENT_DIM)
        generated = model.decode(z).numpy()

    generated_original = scaler.inverse_transform(generated)

    return pd.DataFrame(generated_original, columns=FEATURE_COLUMNS)


# =========================
# VAE QUALITY EVALUATION
# =========================

def evaluate_vae_accuracy(model, scaler, real_df, sample_size=500):
    model.eval()

    real_sample = real_df.head(sample_size)

    scaled_real = scaler.transform(real_sample)
    real_tensor = torch.tensor(scaled_real, dtype=torch.float32)

    with torch.no_grad():
        reconstructed, _, _ = model(real_tensor)

    reconstructed = reconstructed.numpy()
    reconstructed_original = scaler.inverse_transform(reconstructed)

    reconstructed_df = pd.DataFrame(
        reconstructed_original,
        columns=FEATURE_COLUMNS
    )

    mse_per_feature = {}
    similarity_per_feature = {}

    for col in FEATURE_COLUMNS:
        real_values = real_sample[col].values
        reconstructed_values = reconstructed_df[col].values

        mse = np.mean((real_values - reconstructed_values) ** 2)

        real_mean = np.mean(real_values)
        reconstructed_mean = np.mean(reconstructed_values)

        if real_mean != 0:
            similarity = 100 - (abs(real_mean - reconstructed_mean) / abs(real_mean)) * 100
        else:
            similarity = 0

        similarity = max(0, min(100, similarity))

        mse_per_feature[col] = mse
        similarity_per_feature[col] = similarity

    overall_similarity = np.mean(list(similarity_per_feature.values()))

    return reconstructed_df, mse_per_feature, similarity_per_feature, overall_similarity


# =========================
# ANOMALY LOGIC
# =========================

def apply_scenario(row, scenario):
    row = row.copy()

    if scenario == "Normal":
        return row

    if scenario == "Fire Risk":
        row["temperature"] = np.random.uniform(42, 55)
        row["light"] = np.random.uniform(900, 1200)

    elif scenario == "Battery Failure":
        row["voltage"] = np.random.uniform(2.0, 2.6)

    elif scenario == "Humidity Crisis":
        row["humidity"] = np.random.choice([
            np.random.uniform(5, 20),
            np.random.uniform(85, 95)
        ])

    elif scenario == "Random Anomaly":
        anomaly_type = np.random.choice([
            "temperature",
            "humidity",
            "voltage",
            "light"
        ])

        if anomaly_type == "temperature":
            row["temperature"] = np.random.uniform(38, 50)
        elif anomaly_type == "humidity":
            row["humidity"] = np.random.choice([
                np.random.uniform(5, 20),
                np.random.uniform(85, 95)
            ])
        elif anomaly_type == "voltage":
            row["voltage"] = np.random.uniform(2.0, 2.6)
        elif anomaly_type == "light":
            row["light"] = np.random.uniform(850, 1200)

    return row


def calculate_risk_score(row):
    risk = 0
    alerts = []

    if row["temperature"] > 35:
        risk += 30
        alerts.append("High temperature detected")

    if row["humidity"] < 25 or row["humidity"] > 80:
        risk += 25
        alerts.append("Abnormal humidity level")

    if row["light"] > 800:
        risk += 20
        alerts.append("High light intensity")

    if row["voltage"] < 2.8:
        risk += 25
        alerts.append("Low battery voltage")

    risk = min(risk, 100)

    if risk == 0:
        status = "NORMAL"
        message = "System is stable"
    elif risk < 50:
        status = "WARNING"
        message = ", ".join(alerts)
    else:
        status = "CRITICAL"
        message = ", ".join(alerts)

    return risk, status, message


def get_ai_recommendation(status, message):
    if status == "NORMAL":
        return "No action required. Sensor values are within normal range."

    if status == "WARNING":
        return f"AI recommendation: Monitor the system carefully. Detected issue: {message}."

    return f"AI recommendation: Immediate action is required. Critical issue detected: {message}."


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="IoT Monitoring Dashboard",
    layout="wide"
)

st.title("IoT Monitoring Platform")
st.caption("VAE Generative Model + Synthetic IoT Data + AI Anomaly Detection")


with st.spinner("Training VAE model..."):
    model, scaler, real_df, losses = train_vae_model()


st.sidebar.title("Control Panel")

mode = st.sidebar.radio(
    "Choose mode",
    ["Dashboard", "Live Monitoring", "AI Anomaly Detection"]
)

scenario = st.sidebar.selectbox(
    "Demo Scenario",
    [
        "Normal",
        "Random Anomaly",
        "Fire Risk",
        "Battery Failure",
        "Humidity Crisis"
    ]
)

st.sidebar.success("VAE model is ready")


# =========================
# DASHBOARD MODE
# =========================

if mode == "Dashboard":
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌡 Avg Temperature", f"{real_df['temperature'].mean():.2f} °C")
    col2.metric("💧 Avg Humidity", f"{real_df['humidity'].mean():.2f} %")
    col3.metric("💡 Avg Light", f"{real_df['light'].mean():.2f} lx")
    col4.metric("🔋 Avg Voltage", f"{real_df['voltage'].mean():.2f} V")

    synthetic_df = generate_synthetic_data(model, scaler, count=500)

    reconstructed_df, mse_scores, similarity_scores, overall_similarity = evaluate_vae_accuracy(
        model,
        scaler,
        real_df,
        sample_size=500
    )

    st.subheader("VAE Synthetic Data Quality Evaluation")

    acc1, acc2, acc3, acc4, acc5 = st.columns(5)

    acc1.metric("Overall Similarity", f"{overall_similarity:.2f}%")
    acc2.metric("Temperature", f"{similarity_scores['temperature']:.2f}%")
    acc3.metric("Humidity", f"{similarity_scores['humidity']:.2f}%")
    acc4.metric("Light", f"{similarity_scores['light']:.2f}%")
    acc5.metric("Voltage", f"{similarity_scores['voltage']:.2f}%")

    st.subheader("Mean Squared Error by Feature")

    mse_df = pd.DataFrame({
        "Feature": list(mse_scores.keys()),
        "MSE": list(mse_scores.values()),
        "Similarity (%)": list(similarity_scores.values())
    })

    st.dataframe(mse_df)

    st.subheader("Real IoT Data")
    st.dataframe(real_df.head(20))

    st.subheader("Synthetic IoT Data Generated by VAE")
    st.dataframe(synthetic_df.head(20))

    st.subheader("Real vs VAE Reconstructed Data")

    comparison_table = pd.DataFrame({
        "Real Temperature": real_df["temperature"].head(10).values,
        "VAE Temperature": reconstructed_df["temperature"].head(10).values,
        "Real Humidity": real_df["humidity"].head(10).values,
        "VAE Humidity": reconstructed_df["humidity"].head(10).values,
        "Real Light": real_df["light"].head(10).values,
        "VAE Light": reconstructed_df["light"].head(10).values,
        "Real Voltage": real_df["voltage"].head(10).values,
        "VAE Voltage": reconstructed_df["voltage"].head(10).values,
    })

    st.dataframe(comparison_table)

    st.subheader("VAE Training Loss")
    st.line_chart(pd.DataFrame({"loss": losses}))

    st.subheader("Real vs Synthetic Feature Comparison")

    selected_feature = st.selectbox("Select feature", FEATURE_COLUMNS)

    feature_comparison_df = pd.DataFrame({
        "Real": real_df[selected_feature].head(500).reset_index(drop=True),
        "Synthetic": synthetic_df[selected_feature].reset_index(drop=True),
        "Reconstructed": reconstructed_df[selected_feature].reset_index(drop=True)
    })

    st.line_chart(feature_comparison_df)


# =========================
# LIVE MONITORING MODE
# =========================

elif mode == "Live Monitoring":
    st.header("Live Raspberry Pi Monitoring Simulation")

    st.info("This mode simulates Raspberry Pi sending generated sensor data in real time.")

    start = st.button("Start Live Monitoring")

    kpi_placeholder = st.empty()
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    table_placeholder = st.empty()

    history = pd.DataFrame(columns=FEATURE_COLUMNS + ["risk", "status", "message"])

    if start:
        for _ in range(60):
            row = generate_synthetic_data(model, scaler, count=1).iloc[0]
            row = apply_scenario(row, scenario)

            risk, status, message = calculate_risk_score(row)

            history.loc[len(history)] = {
                "temperature": row["temperature"],
                "humidity": row["humidity"],
                "light": row["light"],
                "voltage": row["voltage"],
                "risk": risk,
                "status": status,
                "message": message
            }

            with kpi_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("🌡 Temperature", f"{row['temperature']:.2f} °C")
                c2.metric("💧 Humidity", f"{row['humidity']:.2f} %")
                c3.metric("💡 Light", f"{row['light']:.2f} lx")
                c4.metric("🔋 Voltage", f"{row['voltage']:.2f} V")

            with status_placeholder.container():
                st.subheader("System Status")

                if status == "NORMAL":
                    st.success("🟢 NORMAL — System is stable")
                elif status == "WARNING":
                    st.warning(f"🟡 WARNING — {message}")
                else:
                    st.error(f"🔴 CRITICAL — {message}")

                st.progress(risk / 100)
                st.write(f"Risk Score: **{risk}%**")

            chart_placeholder.line_chart(history[FEATURE_COLUMNS].tail(30))
            table_placeholder.dataframe(history.tail(10))

            time.sleep(1)


# =========================
# AI ANOMALY DETECTION MODE
# =========================

elif mode == "AI Anomaly Detection":
    st.header("AI-Based Anomaly Detection")

    st.info("This mode demonstrates AI alerting based on generated IoT sensor patterns.")

    start = st.button("Start AI Monitoring")

    kpi_placeholder = st.empty()
    alert_placeholder = st.empty()
    chart_placeholder = st.empty()
    table_placeholder = st.empty()

    history = pd.DataFrame(columns=FEATURE_COLUMNS + ["risk", "status", "message"])

    if start:
        for _ in range(60):
            row = generate_synthetic_data(model, scaler, count=1).iloc[0]
            row = apply_scenario(row, scenario)

            risk, status, message = calculate_risk_score(row)
            recommendation = get_ai_recommendation(status, message)

            history.loc[len(history)] = {
                "temperature": row["temperature"],
                "humidity": row["humidity"],
                "light": row["light"],
                "voltage": row["voltage"],
                "risk": risk,
                "status": status,
                "message": message
            }

            with kpi_placeholder.container():
                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric("🌡 Temperature", f"{row['temperature']:.2f} °C")
                c2.metric("💧 Humidity", f"{row['humidity']:.2f} %")
                c3.metric("💡 Light", f"{row['light']:.2f} lx")
                c4.metric("🔋 Voltage", f"{row['voltage']:.2f} V")
                c5.metric("⚠️ Risk", f"{risk}%")

            with alert_placeholder.container():
                st.subheader("AI Alert Center")

                if status == "NORMAL":
                    st.success("🟢 NORMAL")
                elif status == "WARNING":
                    st.warning("🟡 WARNING")
                else:
                    st.error("🔴 CRITICAL ANOMALY DETECTED")

                st.write(f"**Detected message:** {message}")
                st.write(f"**{recommendation}**")
                st.progress(risk / 100)

            chart_placeholder.line_chart(history[FEATURE_COLUMNS].tail(30))
            table_placeholder.dataframe(history.tail(10))

            time.sleep(1)