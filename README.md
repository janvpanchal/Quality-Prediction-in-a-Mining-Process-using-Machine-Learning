# Quality-Prediction-in-a-Mining-Process-using-Machine-Learning
This repository features an end-to-end Machine Learning pipeline designed to function as a Virtual Sensor for an industrial iron ore flotation plant.

Internship project 

Company: UniConverge Technologies Pvt, Ltd (UCT)
Domain: Industrial Automation, Mining, & Predictive Manufacturing

1. Background of the Project
At UniConverge Technologies Pvt, Ltd (UCT), our primary mission is to engineer smart, data-driven automation systems that optimize heavy industrial operations. This project centers on a critical phase of mining logistics: the froth flotation process.
<img width="3999" height="3199" alt="image" src="https://github.com/user-attachments/assets/2aa8aed8-1238-4a52-8d44-54a3690e1e8b" />
In an iron ore flotation plant, raw pulp containing both iron and impurities (primarily silica) is fed into large flotation columns. By injecting controlled air flows and chemical reagents from the bottom, hydrophobic iron particles attach to the air bubbles and float to the top, forming an iron-rich concentrate. Conversely, the unwanted silica sinks and is discharged as waste (tailings).

Maintaining the perfect chemical and mechanical balance inside these columns is vital to maximize iron recovery while keeping silica levels within strict target tolerances.

2. Problem Statement & Relevance
The Challenge
The core metric determining the market value of the output is the % Silica in the final Iron Ore Concentrate. If the silica concentration is too high, the batch fails quality standards.

Currently, the flotation plant relies on physical laboratory sampling to measure this impurity. This lab analysis is slow, returning data only once every hour. In contrast, upstream process variables (such as air flow and pulp levels inside the columns) fluctuate dynamically and are captured by IoT sensors every 20 seconds.
[20-Sec Sensor Data] ---> [ 1-Hour Time Gap ] ---> [ Lab Results (% Silica) ]
  (Air Flow, Levels)        (Blind Spot!)            (Too late for adjustments)

Industrial & Environmental Relevance
Operational Blind Spots: Because of the 1-hour laboratory delay, plant engineers cannot react in real time. If an anomaly occurs at minute 5, it goes undetected until minute 60, resulting in an hour's worth of sub-standard ore or excessive waste.

Environmental Impact: When silica levels spike unexpectedly, operators often over-correct, inadvertently sending vast quantities of valuable iron ore into the tailings dams, wasting resources and filling up tailing structures prematurely.

Solution: By creating an ML-driven predictive proxy, UCT empowers operators with real-time, proactive alerts to make operational adjustments before the ore degrades.

3. System Design
The architecture of our solution is tailored to handle unevenly sampled time-series data and convert it into stable features for a supervised machine learning model.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/e2681d47-1c5d-4f8f-8e05-121421614bdd" />

Core Architecture Components
1. Data Alignment Layer: Resamples the high-frequency 20-second data to uniform 1-minute intervals. 
The hourly target variable (% Silica Concentrate) is forward-filled across the 1-minute steps to match the sensor timestamps during training blocks.
2. Feature Engineering Engine: Extracts temporal features to simulate physical processing delays. Because it takes time for the pulp to move through consecutive flotation cells, we compute 15-minute, 30-minute, and 60-minute lagging states alongside moving averages.
3. Multi-Horizon Forecasting Strategy: Instead of standard autoregressive forecasting (which requires yesterday's outputs to predict today's), we use a Direct Multi-step Approach. Distinct models are trained independently to forecast specific operational windows ahead ($t+60\text{ min}$, $t+120\text{ min}$, etc.).

4. Implementation Details
The core pipeline is written in Python, leveraging LightGBM for high-velocity tree boosting, pandas for advanced time-series transformations, and scikit-learn for strict out-of-sample temporal validation.

5. Results & Core Deliverables
Q1: Is it possible to predict % Silica Concentrate every minute?
Yes. By combining high-frequency sensor readings with forward-filled laboratory targets, the model successfully outputs a new, updated % Silica prediction every single minute. This replaces the 1-hour blind spot with a continuous operational data feed.

Q2: How many steps (hours) ahead can we predict?
Our direct forecasting evaluation showed a clear trade-off between foresight and accuracy:Forecast HorizonEvaluation Metrics (Test Set)Operational Actionability1 Hour Ahead ($t+60$)RMSE: ~0.24%Highly Reliable: Perfect window for manual air flow or setpoint adjustments.2 Hours Ahead ($t+120$)RMSE: ~0.38%Actionable: Good for early warnings regarding feedstock transitions.3 Hours Ahead ($t+180$)RMSE: ~0.55%Degraded: Useful for tracking long-term trends, but less reliable for immediate plant adjustments.

Q3: Can we predict % Silica without using the % Iron Concentrate column?
Yes, and this is standard practice for production deployments. While % Iron Concentrate is highly correlated with % Silica (since they are complementary outputs measured at the end of the line), it is not available in real time. Removing this column entirely and relying strictly on Feed Quality inputs and Flotation Column adjustments keeps the model independent of end-of-pipe lab values. This configuration delivers an stable, predictive proxy using only real-time upstream data.

6. Key Takeaways & Learnings
Domain Insights Over Pure Modeling: In industrial ML systems, you cannot simply throw raw data into an algorithm. Understanding physical delays—specifically, how long it takes ore pulp to travel through successive flotation cells—is critical for building accurate lagging and rolling-window features.
The Danger of Data Leakage: Including % Iron Concentrate as a baseline feature creates a high-performing model on paper, but a useless one in production. Recognizing that both are end-of-process laboratory measurements prevents this kind of data leakage.
Actionable Forecasting Horizons: While long-term predictions are useful, accuracy decays over time. For plant operators, providing a highly precise 1-hour warning window is far more valuable than a less accurate 4-hour projection, giving them a reliable, clear window to step in and optimize performance.
