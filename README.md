📈 HFT Crypto Forecasting Engine (SWT + CNN-GRU)

An enterprise-grade, real-time quantitative trading pipeline that predicts high-frequency cryptocurrency ticks (BTC/USDT) using Continuous Online Learning and Stationary Wavelet Transforms (SWT).

🧠 Project Overview

Traditional machine learning models suffer from "Data Drift" and the "Cold Start" problem. This architecture solves these by implementing Continuous Online Learning. The model never stops training; it learns from live market data every minute, adjusting its weights dynamically to adapt to crypto market volatility.

Key Innovations

Real-Time Streaming: Live ingestion of Binance API data using Apache Kafka.

Signal Denoising: Utilizes Stationary Wavelet Transform (SWT) to filter out high-frequency sensor noise from the market micro-structure.

Hybrid Neural Network: A Conv1D (CNN) layer extracts spatial features, while a Gated Recurrent Unit (GRU) captures long-term temporal momentum.

Time-Travel Queue: A custom data structure that delays training inputs to pair them with true future actuals, completely eliminating the "naive lagging" forecast problem.

HFT Dashboarding: Millisecond-accurate visualization using InfluxDB and Grafana.

🏗️ System Architecture

Collector (realtime_collector.py): Fetches BTC/USDT data from Binance at 1Hz.

Message Broker (Kafka): Buffers and streams data asynchronously.

Pipeline (streaming_pipeline.py): Applies SWT denoising and MinMax scaling.

Brain (online_trainer.py & improved_models.py): Predicts the next market tick and continuously retrains itself on a 300-step rolling memory buffer.

Visualization: InfluxDB stores the time-series data, queried live by Grafana.

🚀 Setup & Installation

1. Start the Infrastructure (Docker)

Ensure Docker is running, then spin up the data layer (Kafka, Zookeeper, InfluxDB, Grafana):

docker-compose up -d


2. Install Python Dependencies

pip install -r requirements.txt


3. Run the Trading Pipeline

Open two separate terminals to run the microservices.

Terminal 1 (Data Ingestion):

python main.py collector


Terminal 2 (AI Worker):

python streaming_worker.py


📊 Grafana Dashboard Setup

Navigate to http://localhost:3000 (admin/admin).

Add InfluxDB as a Data Source (Flux language, Bucket: forecast_bucket).

Import the Quant Trading panels to monitor Live Prediction Delta, API Latency, and the AI Forecast vs. Actuals overlap.

🎓 Academic Context

This project was developed as a comprehensive thesis on applying advanced signal processing (Wavelets) and adaptive deep learning to non-stationary financial time series data.