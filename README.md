# IoT-GAN Monitoring Platform

## Project Title

IoT-GAN Monitoring Platform: Synthetic IoT Sensor Data Generation and AI-Based Anomaly Detection

## Project Objective

The objective of this project is to develop a web-based IoT monitoring platform that uses a Generative Adversarial Network (GAN) to generate realistic synthetic sensor data and demonstrate intelligent anomaly detection for IoT environments.

The platform simulates a Raspberry Pi-based monitoring system and provides real-time visualization of generated sensor values, system status monitoring, anomaly detection, and AI recommendations.

## Project Relevance

Internet of Things (IoT) systems generate large volumes of sensor data that are required for monitoring, analytics, and machine learning applications. In many real-world scenarios, obtaining sufficient datasets can be difficult due to hardware limitations, privacy concerns, or deployment costs.

Generative Adversarial Networks (GANs) provide an effective solution by generating realistic synthetic sensor data that can be used for system testing, research, simulation, and machine learning model training.

## Technologies Used

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Dataset Description

The project uses a synthetic IoT dataset containing the following sensor parameters:

| Sensor | Description |
|----------|----------|
| Temperature | Ambient temperature (°C) |
| Humidity | Relative humidity (%) |
| Light | Light intensity (lux) |
| Voltage | Battery voltage (V) |

Dataset size:

- 3000 records
- 4 sensor features

## GAN Architecture

### Generator

The Generator receives a random noise vector and produces synthetic IoT sensor values.

Architecture:

```text
Noise (16)
    ↓
Linear(16 → 64)
    ↓
BatchNorm
    ↓
ReLU
    ↓
Linear(64 → 128)
    ↓
BatchNorm
    ↓
ReLU
    ↓
Linear(128 → 4)
    ↓
Sigmoid
```

### Discriminator

The Discriminator receives sensor data and determines whether the sample is real or generated.

Architecture:

```text
Input (4)
    ↓
Linear(4 → 128)
    ↓
LeakyReLU
    ↓
Dropout
    ↓
Linear(128 → 64)
    ↓
LeakyReLU
    ↓
Dropout
    ↓
Linear(64 → 1)
    ↓
Sigmoid
```

## Training Parameters

| Parameter | Value |
|------------|------------|
| Epochs | 300 |
| Batch Size | 64 |
| Learning Rate | 0.0002 |
| Noise Dimension | 16 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy (BCE) |

## Project Functionality

### Dashboard Mode

The dashboard provides:

- Dataset overview
- Average sensor statistics
- Generated synthetic data
- Real vs Fake comparison
- GAN quality evaluation
- Training loss visualization
- CSV export functionality

### Live Monitoring Mode

This mode simulates Raspberry Pi real-time monitoring:

- Temperature monitoring
- Humidity monitoring
- Light monitoring
- Voltage monitoring
- System status updates
- Risk score calculation
- Historical sensor charts

### AI Anomaly Detection Mode

The AI module analyzes generated sensor values and detects abnormal situations.

Supported scenarios:

- Normal operation
- Fire Risk
- Battery Failure
- Humidity Crisis
- Random Anomaly

The system generates:

- Risk score
- Alert status
- Diagnostic message
- AI recommendation

## Anomaly Detection Rules

### High Temperature

Condition:

```text
Temperature > 35°C
```

Risk contribution:

```text
+30 points
```

### Humidity Crisis

Condition:

```text
Humidity < 25%
OR
Humidity > 80%
```

Risk contribution:

```text
+25 points
```

### High Light Intensity

Condition:

```text
Light > 800 lux
```

Risk contribution:

```text
+20 points
```

### Battery Failure

Condition:

```text
Voltage < 2.8V
```

Risk contribution:

```text
+25 points
```

Maximum risk score:

```text
100 points
```

## Evaluation Metrics

The project evaluates GAN quality using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Feature Similarity (%)

Overall similarity is calculated as the average similarity across all sensor features.

## Results

The trained GAN successfully generates realistic IoT sensor data that follows the statistical distribution of the original dataset.

The platform demonstrates:

- Successful adversarial training
- Realistic synthetic data generation
- Real-time monitoring simulation
- AI-based anomaly detection
- Interactive visualization through Streamlit

## Project Structure

```text
iot-gan-monitoring-platform/
│
├── app.py
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── generated_data.csv
│   ├── monitoring_log.csv
│   └── gan_model.pth
│
└── screenshots/
```

## Installation

Clone repository:

```bash
git clone https://github.com/your-username/iot-gan-monitoring-platform.git
```

Move into project folder:

```bash
cd iot-gan-monitoring-platform
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start Streamlit application:

```bash
streamlit run app.py
```

Application URL:

```text
http://localhost:8501
```

## Future Work

Future improvements include:

- Integration with real Raspberry Pi sensors
- MQTT communication protocol support
- PostgreSQL database storage
- WebSocket real-time updates
- Cloud deployment
- WGAN-GP implementation for improved stability
- Advanced anomaly detection using deep learning
- Multi-device monitoring support

## Conclusion

This project demonstrates the practical application of Generative Adversarial Networks for synthetic IoT data generation and intelligent monitoring. The developed system successfully combines GAN-based data generation, anomaly detection, and real-time visualization within a web-based monitoring platform. The solution can serve as a foundation for future IoT monitoring systems based on Raspberry Pi and other embedded platforms.

## Author

Yessentay Kurmanbay

Master's Program in Computer Science and Software Engineering (CSSE)

International Information Technology University (IITU)