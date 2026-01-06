# 📈 Stock Agent Ops - Complete Project Analysis

## Table of Contents
1. [Project Overview](#project-overview)
2. [Where to Start](#where-to-start)
3. [Architecture & Tech Stack](#architecture--tech-stack)
4. [Detailed Component Analysis](#detailed-component-analysis)
5. [Data Flow](#data-flow)
6. [File-by-File Breakdown](#file-by-file-breakdown)

---

## Project Overview

**Stock Agent Ops** is a **production-grade MLOps pipeline** that automates end-to-end stock market analysis and prediction. It combines:

- **Transfer Learning (LSTM Neural Networks)**: Parent-Child architecture where a base model trained on S&P 500 (^GSPC) is fine-tuned for individual stocks
- **Agentic AI (LangGraph)**: Multi-agent system acting as financial analysts generating Bloomberg-quality reports
- **Feature Store (Feast)**: Centralized feature management for both training and serving
- **Observability Stack**: Prometheus + Grafana for monitoring, Evidently AI for drift detection
- **Real-time Serving**: FastAPI with Redis caching for low-latency predictions

### Key Value Proposition
The system can predict individual stock prices **with minimal historical data** by leveraging transfer learning from a pre-trained parent model trained on market-wide data.

---

## Where to Start

### Step 1: Understand the Overall Architecture
Read these files **in order**:
1. **[README.md](README.md)** - High-level overview and setup
2. **[doc/system_design.md](doc/system_design.md)** - Detailed technical architecture
3. **[src/config.py](src/config.py)** - Configuration constants

### Step 2: Understand the Data Flow
1. Data ingestion: [src/data/ingestion.py](src/data/ingestion.py)
2. Data preparation: [src/data/preparation.py](src/data/preparation.py)
3. Feature store: [feature_store/feature_store.yaml](feature_store/feature_store.yaml)

### Step 3: Understand Model Training
1. Model definition: [src/model/definition.py](src/model/definition.py)
2. Training pipeline: [src/pipelines/training_pipeline.py](src/pipelines/training_pipeline.py)
3. Evaluation: [src/model/evaluation.py](src/model/evaluation.py)

### Step 4: Understand Inference
1. Inference pipeline: [src/pipelines/inference_pipeline.py](src/pipelines/inference_pipeline.py)
2. Predictions: [src/inference.py](src/inference.py)

### Step 5: Understand Agents
1. Agent graph: [src/agents/graph.py](src/agents/graph.py)
2. Agent nodes: [src/agents/nodes.py](src/agents/nodes.py)
3. Agent tools: [src/agents/tools.py](src/agents/tools.py)

### Step 6: Understand Monitoring
1. Drift detection: [src/monitoring/drift.py](src/monitoring/drift.py)
2. Agent evaluation: [src/monitoring/agent_eval.py](src/monitoring/agent_eval.py)

### Step 7: Understand API & Frontend
1. Backend API: [backend/api.py](backend/api.py)
2. Frontend UI: [frontend/app.py](frontend/app.py)

---

## Architecture & Tech Stack

### 🏗️ System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER LAYER                                │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  Streamlit UI    │              │  Grafana Monitor │         │
│  │  (frontend/)     │              │  Dashboard       │         │
│  └────────┬─────────┘              └────────┬─────────┘         │
└───────────┼─────────────────────────────────┼───────────────────┘
            │                                  │
┌───────────┼──────────────────────────────────┼───────────────────┐
│           │        ORCHESTRATION LAYER       │                   │
│  ┌────────▼─────────────────────────────────▼────────┐           │
│  │         FastAPI (backend/main.py)                │           │
│  │  - Routes & Endpoints (backend/api.py)          │           │
│  │  - Rate Limiting & Caching                      │           │
│  │  - Task Management & Status                     │           │
│  └────────┬─────────────────────────────────────────┘           │
└───────────┼────────────────────────────────────────────────────┬┘
            │                                                     │
┌───────────┼─────────────────────────────────────────┬──────────┼────┐
│           │      ML PIPELINE LAYER                  │          │    │
│  ┌────────▼────────────┐  ┌───────────────────┐   │          │    │
│  │ Training Pipeline   │  │ Inference Pipeline│   │          │    │
│  │ - Data Ingestion    │  │ - Load Models     │   │          │    │
│  │ - Data Prep         │  │ - Run Inference   │   │          │    │
│  │ - Model Training    │  │ - Return Preds    │   │          │    │
│  │ - Evaluation        │  └───────────────────┘   │          │    │
│  └────────┬────────────┘                          │          │    │
│           │                                       │          │    │
│  ┌────────▼────────────────────────────────────┐  │          │    │
│  │     Agent System (LangGraph)               │  │          │    │
│  │ 1. Performance Analyst Node ──────────┐    │  │          │    │
│  │ 2. Market Expert Node ──────────┐     │    │  │          │    │
│  │ 3. Report Generator Node ──────┐│     │    │  │          │    │
│  │ 4. Critic Node ────────────────┘│     │    │  │          │    │
│  │ ↑ Semantic Cache (Qdrant) ──────┤─────┤    │  │          │    │
│  └────────┬────────────────────────┼────┘    │  │          │    │
│           │                        │         │  │          │    │
└───────────┼────────────────────────┼─────────┼──┼─────┬─────┼────┘
            │                        │         │  │     │     │
┌───────────┼────────────────────────┼─────────┼──┼─────┼─────┼────┐
│           │    STORAGE & CACHE    │         │  │     │     │    │
│  ┌────────▼─────────┐  ┌──────────▼──┐     │  │     │     │    │
│  │  Redis Stack     │  │ Qdrant Vector│     │  │     │     │    │
│  │  - Data Cache    │  │ Database     │     │  │     │     │    │
│  │  - Session Store │  │ (Semantic    │     │  │     │     │    │
│  │  - Task Queue    │  │  Cache)      │     │  │     │     │    │
│  └──────────────────┘  └──────────────┘     │  │     │     │    │
│                                              │  │     │     │    │
│  ┌──────────────────────────────────────┐   │  │     │     │    │
│  │   Feast Feature Store                │   │  │     │     │    │
│  │   (feature_store/feature_store.yaml) │   │  │     │     │    │
│  │   - Training Features                │   │  │     │     │    │
│  │   - Serving Features                 │   │  │     │     │    │
│  └──────────────────────────────────────┘   │  │     │     │    │
│                                              │  │     │     │    │
│  ┌──────────────────────────────────────┐   │  │     │     │    │
│  │   MLflow (DagsHub)                   │   │  │     │     │    │
│  │   - Experiment Tracking              │   │  │     │     │    │
│  │   - Model Registry                   │   │  │     │     │    │
│  └──────────────────────────────────────┘   │  │     │     │    │
└──────────────────────────────────────────────┼──┼─────┼─────┼────┘
                                               │  │     │     │
┌──────────────────────────────────────────────┼──┼─────┼─────┼────┐
│           OBSERVABILITY LAYER               │  │     │     │    │
│  ┌──────────────────┐  ┌─────────────────┐ │  │     │     │    │
│  │   Prometheus     │  │  Drift Monitor  │ │  │     │     │    │
│  │   - Metrics      │  │  - Data Drift   │ │  │     │     │    │
│  │   - Scraping     │  │  - Model Drift  │ │  │     │     │    │
│  └──────────────────┘  └─────────────────┘ │  │     │     │    │
│  └────────────────────────────────────────┘ │  │     │     │    │
│        System Metrics & Health Checks      │  │     │     │    │
└──────────────────────────────────────────────┴──┴─────┴─────┴────┘
```

### 📦 Tech Stack

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Model** | Deep Learning Framework | PyTorch (LSTM) |
| **LLM** | Language Model | Ollama (gpt-oss:20b-cloud) |
| **Embeddings** | Vector Embeddings | Ollama (nomic-embed-text) |
| **AI Framework** | Agent Orchestration | LangGraph + LangChain |
| **Feature Store** | Feature Management | Feast |
| **Registry** | Experiment Tracking | MLflow (DagsHub) |
| **Vector DB** | Semantic Storage | Qdrant |
| **Cache** | In-Memory Store | Redis Stack |
| **Backend** | API Server | FastAPI + Uvicorn |
| **Frontend** | UI Framework | Streamlit |
| **Monitoring** | Time-Series DB | Prometheus |
| **Dashboard** | Visualization | Grafana |
| **Orchestration** | Container Runtime | Docker Compose / Kubernetes |

---

## Detailed Component Analysis

### 1. Configuration Layer (`src/config.py`)

**Purpose**: Centralized configuration for the entire system

**Key Variables**:
```python
device = "cuda" or "cpu"          # Computation device
context_len = 60                   # Historical lookback window (days)
pred_len = 5                       # Prediction horizon (days)
features = [7 technical features]  # Input feature list (Open, High, Low, Close, Volume, RSI14, MACD)
batch_size = 32                    # Training batch size
parent_ticker = "^GSPC"            # S&P 500 base model
child_tickers = [list]             # Individual stocks to predict (GOOG, AMZN, META, TSLA)
parent_epochs = 20                 # Parent training epochs
child_epochs = 10                  # Child fine-tuning epochs
transfer_strategy = "freeze"       # Transfer learning: freeze or fine_tune
```

**Why It Matters**:
- Single source of truth for all hyperparameters
- Easy to experiment with different configurations
- Ensures consistency across training, inference, and monitoring

---

### 2. Data Ingestion Layer (`src/data/ingestion.py`)

**Purpose**: Fetch raw market data and compute technical indicators

**Key Functions**:

**`fetch_ohlcv(ticker, start, end)`**:
- Downloads historical price data using yfinance
- Calculates technical indicators:
  - **RSI (Relative Strength Index)**: Momentum oscillator (14-period)
  - **MACD (Moving Average Convergence Divergence)**: Trend following
- Validates data quality (no NaNs, sufficient length)
- **Feast Integration**: Saves data to Parquet for feature store
  - Adds `ticker`, `event_timestamp`, `created_timestamp` columns
  - Handles concurrent writes with file locks

**Technical Indicators Explained**:
- **RSI14**: Values 0-100, signals overbought (>70) or oversold (<30)
- **MACD**: Difference between 12-day and 26-day EMA, signals momentum

**Data Validation**:
```
✓ Data length >= context_len (60) + pred_len (5) = 65 days minimum
✓ No NaN values in features
✓ All numeric types
```

---

### 3. Data Preparation Layer (`src/data/preparation.py`)

**Purpose**: Convert raw data into ML-ready tensors

**Key Class**: `StockDataset` (PyTorch Dataset)
- Normalizes features using `StandardScaler`
- Creates sliding windows of size `context_len=60`
- Returns (X, y) pairs where:
  - **X**: Historical 60 days of features → shape `(60, 7)`
  - **y**: Next 5 days of features → shape `(5, 7)`

**Example**:
```
Day 1-60: Historical OHLCV + RSI + MACD → Input X
Day 61-65: Actual OHLCV + RSI + MACD → Target y

Model learns: X (60 days) → predict y (5 days)
```

---

### 4. Model Architecture (`src/model/definition.py`)

**Purpose**: Define the LSTM neural network

**Model: `LSTMModel`**
```python
LSTM (Recurrent Neural Network) Pipeline:
┌────────────────────────────────────┐
│ Input: (batch, 60, 7)              │  60 timesteps, 7 features
├────────────────────────────────────┤
│ LSTM Layer 1 (128 hidden units)    │  3 stacked layers
│ Dropout: 0.2                       │  Prevents overfitting
├────────────────────────────────────┤
│ LSTM Layer 2 (128 hidden units)    │
│ LSTM Layer 3 (128 hidden units)    │
├────────────────────────────────────┤
│ Fully Connected: 128 → 35          │  35 = 5 days × 7 features
├────────────────────────────────────┤
│ Output: (batch, 5, 7)              │  5-day forecast, 7 features
└────────────────────────────────────┘
```

**Why LSTM?**
- Captures temporal dependencies in sequential data
- Remembers patterns from 60 days of history
- Better than simple linear models for non-linear market patterns

---

### 5. Training Pipeline (`src/pipelines/training_pipeline.py`)

**Purpose**: Train models from raw data to deployed artifact

**Two-Stage Training**:

#### Stage 1: Parent Model Training (`train_parent()`)
- **Data**: S&P 500 (^GSPC) from 2004-present (20 years)
- **Purpose**: Learn general market patterns
- **Epochs**: 20
- **Output**: `outputs/parent/^GSPC_parent_model.pt`
- **Tracked in MLflow**: Hyperparameters, metrics, artifacts

**What the parent learns**:
- How market indices behave
- General momentum and trend patterns
- Volatility characteristics
- Seasonal effects

#### Stage 2: Child Model Training (`train_child(ticker)`)
- **Data**: Individual stock (e.g., GOOG) from inception
- **Transfer Strategy**: Freeze parent weights, add small fine-tuning layer
- **Epochs**: 10 (less data = less overfitting risk)
- **Output**: `outputs/{TICKER}/{TICKER}_child_model.pt`
- **Advantage**: Requires less data, learns faster

**Transfer Learning Benefits**:
```
Without Transfer Learning:
- Need 5+ years of stock data
- Training takes days
- High variance with limited samples

With Transfer Learning:
- Works with 1-2 years of data
- Training takes hours
- Leverages general market knowledge from parent
```

**MLflow Tracking**:
```
Logged Metrics:
- train_loss (per epoch)
- val_loss
- train_time
- val_rmse (prediction error)
- val_mape (percentage error)
```

---

### 6. Inference Pipeline (`src/pipelines/inference_pipeline.py` & `src/inference.py`)

**Purpose**: Generate predictions using trained models

**Functions**:

**`predict_parent()`**:
- Loads parent model from `outputs/parent/^GSPC_parent_model.pt`
- Loads parent scaler
- Returns S&P 500 forecast

**`predict_child(ticker)`**:
- Checks if child model exists
- If missing → triggers automatic training (background task)
- Returns: **"__MODEL_TRAINING__"** status while training
- Loads child model once ready
- Returns stock forecast

**`predict_one_step_and_week()`**:
- **Input**: Last 60 days of OHLCV data
- **Inference**: Forward pass through LSTM
- **Denormalization**: Inverse transform from StandardScaler
- **Output Structure**:
```json
{
  "ticker": "GOOG",
  "last_date": "2024-01-15",
  "next_business_days": ["2024-01-16", "2024-01-17", ...],
  "predictions": {
    "next_day": {"date": "2024-01-16", "open": 180.5, "high": 182.3, ...},
    "next_week": {"high": 185.2, "low": 179.1},
    "full_forecast": [5 daily forecasts]
  }
}
```

---

### 7. Feature Store (`feature_store/`)

**Purpose**: Centralized feature management for training and serving

**Architecture**:
```
feature_store/
├── feature_store.yaml    # Feast config (tickers, features, data source)
├── features.py           # Feature definitions
└── data/features.parquet # Raw feature data
```

**Feast Integration**:
1. **Data Source**: Parquet file with columns `[ticker, event_timestamp, OHLCV, RSI, MACD]`
2. **Feature View**: Define which features to expose (e.g., "close_price", "rsi14")
3. **Materialization**: Store features in online store (Redis) for low-latency serving
4. **Benefits**:
   - Single source of truth for features
   - Prevents training-serving skew
   - Can version features
   - Easy to add new features

---

### 8. Agent System (`src/agents/`)

**Purpose**: Multi-agent system that generates financial analysis reports

**Architecture**: LangGraph State Machine

```
                  ┌──────────────────────┐
                  │  Input: Ticker       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Performance Analyst  │
                  │ - Analyze predictions│
                  │ - Identify trends    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Market Expert       │
                  │ - Fetch news sentiment
                  │ - Analyze macro      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Report Generator    │
                  │ - Combine insights   │
                  │ - Create Bloomberg   │
                  │   style report       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Critic (QA)         │
                  │ - Validate accuracy  │
                  │ - Check consistency  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Final Report + Rec   │
                  └──────────────────────┘
```

**Key Components**:

**`src/agents/graph.py`**:
- **`AgentState`**: Shared state machine
  - `ticker`: Stock symbol
  - `predictions`: Model forecast data
  - `news_sentiment`: Market sentiment
  - `final_report`: Generated analysis
  - `recommendation`: Buy/Sell/Hold
  - `confidence`: Analysis confidence

- **`build_graph()`**: Constructs LangGraph workflow
- **`analyze_stock(ticker)`**: Entry point, handles semantic caching

**Semantic Caching**:
```python
1. User requests: "Analysis for GOOG"
2. Embed query: "Analysis report for GOOG" → vector
3. Search Qdrant: Find similar cached reports
4. If similarity > 0.95 and same ticker → Return cached result
5. Otherwise → Run full agent pipeline
6. Cache result for future requests
```

**`src/agents/nodes.py`**:
- **Performance Analyst Node**: Analyzes 5-day forecast trends
- **Market Expert Node**: Fetches news and sentiment using tools
- **Report Generator Node**: Combines all insights into report
- **Critic Node**: Quality assurance and validation

**`src/agents/tools.py`**:
- **`get_stock_predictions(ticker)`**: Calls inference API
- **`get_stock_news(ticker)`**: Fetches news from external API
- Agents can invoke these tools autonomously

**LLM**: Ollama `gpt-oss:20b-cloud` (open-source model)

---

### 9. Monitoring & Observability

#### 9.1 Drift Detection (`src/monitoring/drift.py`)

**Purpose**: Detect when models degrade due to changing market conditions

**Two Types of Drift**:

**1. Data Drift** (Feature Distribution Shift):
```
Reference Period: Last 180 days of historical data
Current Period:   Last 30 days of recent data

For each feature (Open, High, Low, Close, Volume):
  Z-Score = |current_mean - reference_mean| / reference_std
  
  If Z-Score > 2.0 → Data distribution has shifted significantly
  Model assumes old patterns, now seeing new patterns → Performance drops
```

**2. Concept Drift** (Market Behavior Shift):
```
Volatility Index = current_volatility / reference_volatility

If vol_ratio > 2.5 → Market is 2.5x more volatile
If vol_ratio < 0.4 → Market is 60% less volatile

Example: Fed rate hike → volatility spikes → models need retraining
```

**Health Status**:
```
Healthy:           drift_score < 1.0, vol_ratio 0.6-1.5
Degraded Warning:  drift_score 1.0-2.0, vol_ratio 1.5-2.5
Critical:          drift_score > 2.0, vol_ratio > 2.5 or < 0.4

Action: When Critical → Trigger automatic retraining
```

#### 9.2 Agent Evaluation (`src/monitoring/agent_eval.py`)

**Purpose**: Evaluate quality of generated reports

**Metrics**:
- **Coherence**: Are multiple agents saying consistent things?
- **Accuracy**: Do recommendations match prediction trends?
- **Recency**: Is report based on latest data?
- **Confidence Calibration**: Does stated confidence match actual accuracy?

---

### 10. Backend API (`backend/api.py`)

**Purpose**: REST API to expose all system functionality

**Architecture**: FastAPI (async)

**Key Endpoints**:

#### Training Endpoints
- `POST /train-parent` → Train S&P 500 base model
- `POST /train-child` → Train individual stock (background task)
- `GET /status/{task_id}` → Check training progress

#### Prediction Endpoints
- `POST /predict-parent` → Forecast S&P 500
- `POST /predict-child` → Forecast specific stock (auto-train if needed)
- Response includes: predictions, recommendations, confidence

#### Monitoring Endpoints
- `POST /monitor/parent` → Check parent model health
- `POST /monitor/{ticker}` → Check ticker health
- `GET /monitor/{ticker}/drift` → Get drift report (JSON)
- `GET /monitor/{ticker}/eval` → Get agent evaluation (JSON)

#### System Endpoints
- `GET /` → API info & available commands
- `GET /health` → Health check
- `GET /system/cache` → Inspect Redis
- `GET /system/logs` → Retrieve logs
- `DELETE /system/reset` → Wipe all data
- `GET /metrics` → Prometheus metrics

**Caching Strategy**:
```
Request comes in for GOOG prediction:
┌─────────────────────────────────┐
│ 1. Check Redis Cache            │
│    Key: "pred:GOOG"             │
├─────────────────────────────────┤
│ If cache HIT:                   │
│ ✓ Check age (TTL = 1 day)       │
│ ✓ Return cached result          │
│ (99% of requests hit cache)     │
└─────────────────────────────────┘
        │
        NO HIT
        │
        ▼
┌─────────────────────────────────┐
│ 2. Run Inference                │
│    - Load model                 │
│    - Fetch latest data          │
│    - Run prediction             │
│    - Save to Redis (1-day TTL)  │
└─────────────────────────────────┘
```

**Rate Limiting**:
- Max 100 requests per minute per IP
- Prevents abuse of expensive inference

**Background Tasks** (Redis Queue):
- Training jobs run asynchronously
- User gets `task_id` to check status
- System automatically trains missing models

---

### 11. Frontend UI (`frontend/app.py`)

**Purpose**: User-friendly interface for analysis

**Built with**: Streamlit (Python web framework)

**Features**:

**Input Section**:
- Ticker symbol input (e.g., "AAPL", "GOOG")
- "Generate Analysis" button
- Session tracking

**Output Sections**:
1. **Prediction Card**: 5-day forecast chart
   - Next-day price prediction
   - Weekly high/low range
   - Full forecast with confidence intervals

2. **Report Card**: Agent-generated narrative
   - Market expert sentiment
   - Performance analysis
   - Bloomberg-style recommendation

3. **Metrics Card**: Key statistics
   - Recommendation: Buy/Sell/Hold/Neutral
   - Confidence: High/Medium/Low
   - Trending direction

4. **Monitoring**: System health
   - Drift status
   - Last update time
   - Cache hit rate

**Data Visualization**:
- Plotly interactive charts
- Dark theme (professional)
- Responsive layout

---

### 12. Containerization & Orchestration

#### Docker Compose (`docker-compose.yml`)

**Services**:

1. **FastAPI** (Backend)
   - Port 8000
   - Mounts: `/outputs`, `/mlruns`, `/feature_store`
   - Depends on: Redis
   - Workers: 4

2. **Redis Stack**
   - Port 6379 (data)
   - Port 8001 (UI)
   - Persistent volume: `redis_data`

3. **Qdrant** (Vector DB)
   - Port 6333
   - Persistent volume: `qdrant_data`
   - For semantic caching

4. **Prometheus** (Metrics)
   - Port 9090
   - Scrapes FastAPI metrics
   - Config: `prometheus/prometheus.yml`

5. **Grafana** (Dashboard)
   - Port 3000
   - Default user: admin/admin
   - Visualizes Prometheus data

**Networking**: All services on `app_network` bridge network

#### Kubernetes (`k8s/`)

**Files**:
- `fastapi.yaml` → Deployment + Service
- `redis.yaml` → StatefulSet with PVC
- `qdrant.yaml` → Deployment
- `prometheus.yaml` → ConfigMap + Deployment
- `grafana.yaml` → Deployment + Service
- `volumes.yaml` → PersistentVolumes

**Scaling**: Can adjust replicas for high availability

---

### 13. MLflow & Experiment Tracking

**Purpose**: Version control for models and experiments

**What Gets Tracked**:
```
training_run:
├── Parameters
│  ├── context_len: 60
│  ├── pred_len: 5
│  ├── batch_size: 32
│  ├── epochs: 20
│  └── ...
├── Metrics
│  ├── train_loss: [values per epoch]
│  ├── val_loss
│  ├── val_rmse: 2.45
│  └── val_mape: 3.2%
├── Artifacts
│  ├── model.pt
│  ├── scaler.pkl
│  └── config.yaml
└── Tags
   ├── model_type: "parent"
   ├── ticker: "^GSPC"
   └── status: "Production"
```

**Registry**: DagsHub MLflow (hosted)
- Models stored remotely
- Version history
- Deployment staging

---

## Data Flow

### Complete End-to-End Flow

```
User Action: "Predict GOOG"
│
├─ 1. CHECK CACHE
│  ├─ Redis lookup: key="pred:GOOG"
│  ├─ If HIT (age < 1 day) → Return cached
│  └─ If MISS → Continue
│
├─ 2. FETCH DATA
│  ├─ Download last 60 days of GOOG price data (yfinance)
│  ├─ Calculate RSI14, MACD indicators
│  └─ Validate data quality
│
├─ 3. CHECK MODEL
│  ├─ Does GOOG_child_model.pt exist?
│  ├─ If NO → Queue training task (background)
│  │   ├─ Load parent model (^GSPC_parent_model.pt)
│  │   ├─ Fine-tune on GOOG data
│  │   ├─ Save child model
│  │   └─ Return status: "__MODEL_TRAINING__"
│  └─ If YES → Continue
│
├─ 4. PREPARE DATA FOR INFERENCE
│  ├─ Normalize using StandardScaler
│  ├─ Create sliding window X: (1, 60, 7)
│  └─ X = last 60 days of features
│
├─ 5. RUN INFERENCE
│  ├─ Load GOOG child model
│  ├─ Model(X) → output: (1, 5, 7)
│  ├─ Denormalize predictions
│  └─ Get 5-day forecast
│
├─ 6. RUN AGENTS (if NOT in semantic cache)
│  │
│  ├─ Agent 1: Performance Analyst
│  │  └─ Analyzes 5-day trend
│  │
│  ├─ Agent 2: Market Expert
│  │  ├─ Fetches latest GOOG news
│  │  └─ Summarizes sentiment
│  │
│  ├─ Agent 3: Report Generator
│  │  ├─ Combines all insights
│  │  └─ Generates Bloomberg-style report
│  │
│  └─ Agent 4: Critic
│     ├─ Validates coherence
│     └─ Outputs recommendation + confidence
│
├─ 7. CACHE RESULT
│  ├─ Store in Redis (1-day TTL)
│  ├─ Store in Qdrant (semantic search)
│  └─ Log to MLflow (tracking)
│
├─ 8. RETURN RESPONSE
│  └─ {
│      "predictions": {...},
│      "final_report": "...",
│      "recommendation": "Buy",
│      "confidence": "High"
│     }
│
└─ 9. MONITOR & DRIFT CHECK
   ├─ Periodically check data drift
   ├─ Evaluate agent quality
   └─ If drift detected → Trigger retraining

```

---

## File-by-File Breakdown

### Root Level Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, redirects to `backend.main:app` |
| `pyproject.toml` | Project metadata & dependencies |
| `README.md` | Project overview & setup guide |
| `.env` | Environment variables (credentials, API keys) |
| `.python-version` | Python version specification |
| `docker-compose.yml` | Multi-container orchestration |
| `run_docker.sh` | Script to start Docker services |
| `run_k8s.sh` | Script to deploy to Kubernetes |

### `src/` - Core ML Logic

| File | Purpose |
|------|---------|
| `config.py` | **Hyperparameters & settings** |
| `exception.py` | Custom exception classes |
| `utils.py` | Utility functions (MLflow setup, dir init) |
| `inference.py` | **Prediction inference wrapper** |

### `src/agents/` - AI Agent System

| File | Purpose |
|------|---------|
| `graph.py` | **LangGraph workflow orchestration** |
| `nodes.py` | **Agent node implementations** (4 agents) |
| `tools.py` | Tools agents can invoke (predictions, news) |

### `src/data/` - Data Pipeline

| File | Purpose |
|------|---------|
| `ingestion.py` | **Fetch OHLCV + technical indicators** |
| `preparation.py` | Convert raw data to ML tensors |

### `src/model/` - Model Definition & Training

| File | Purpose |
|------|---------|
| `definition.py` | **LSTM model architecture** |
| `training.py` | Model training loop (forward pass, backprop) |
| `evaluation.py` | Metrics calculation (RMSE, MAPE) |
| `saving.py` | Model checkpointing |

### `src/pipelines/` - ML Workflows

| File | Purpose |
|------|---------|
| `training_pipeline.py` | **End-to-end training (parent + child)** |
| `inference_pipeline.py` | End-to-end prediction |

### `src/monitoring/` - Model Monitoring

| File | Purpose |
|------|---------|
| `drift.py` | **Data/concept drift detection** |
| `agent_eval.py` | Agent report quality evaluation |

### `src/memory/` - Semantic Memory

| File | Purpose |
|------|---------|
| `semantic_cache.py` | Qdrant integration for caching |

### `backend/` - API Server

| File | Purpose |
|------|---------|
| `main.py` | **FastAPI app initialization** |
| `api.py` | **All REST endpoints** (100+ lines each route) |
| `schemas.py` | Pydantic request/response schemas |
| `state.py` | Global state (Redis client, metrics) |
| `tasks.py` | Background tasks (async training) |
| `rate_limiter.py` | Rate limiting middleware |
| `Dockerfile` | Container image spec |
| `requirements.txt` | Python dependencies |

### `frontend/` - User Interface

| File | Purpose |
|------|---------|
| `app.py` | **Streamlit UI** (entire frontend) |
| `Dockerfile` | Container spec |
| `requirements.txt` | Dependencies |

### `logger/` - Logging

| File | Purpose |
|------|---------|
| `logger.py` | Centralized logging setup |
| `__init__.py` | Module initialization |

### `feature_store/` - Feast Feature Store

| File | Purpose |
|------|---------|
| `feature_store.yaml` | Feast configuration |
| `features.py` | Feature definitions |
| `data/features.parquet` | Raw feature data |

### `prometheus/` & `grafana/` - Observability

| File | Purpose |
|------|---------|
| `prometheus.yml` | Prometheus config (scrape targets) |
| `grafana-dashboard.json` | Pre-built Grafana dashboard |

### `k8s/` - Kubernetes Manifests

| File | Purpose |
|------|---------|
| `fastapi.yaml` | FastAPI deployment |
| `redis.yaml` | Redis StatefulSet |
| `qdrant.yaml` | Qdrant deployment |
| `prometheus.yaml` | Prometheus deployment |
| `grafana.yaml` | Grafana deployment |
| `volumes.yaml` | Persistent volumes |

### `doc/` - Documentation

| File | Purpose |
|------|---------|
| `system_design.md` | Deep technical architecture |
| `AWS.md` | AWS deployment guide |
| `k8s.md` | Kubernetes setup guide |
| `commands.md` | CLI command reference |

### `outputs/` - Model Artifacts

| Directory | Purpose |
|-----------|---------|
| `parent/` | Parent model files (^GSPC_parent_model.pt) |
| `{TICKER}/` | Child model files per stock |

### `mlruns/` & `mlartifacts/` - MLflow Tracking

Stores experiment runs, metrics, and model artifacts

---

## Key Concepts Summary

### Transfer Learning Strategy
1. **Parent Model** (Large, General):
   - Trained on 20 years of S&P 500 data
   - Learns universal market patterns
   - Takes longest to train (20 epochs)

2. **Child Model** (Small, Specific):
   - Starts from parent weights
   - Fine-tuned on individual stock data
   - Requires minimal stock history
   - Trains quickly (10 epochs)
   - More accurate with limited data

### Feature Engineering
- **Raw Features**: Open, High, Low, Close, Volume
- **Technical Indicators**:
  - RSI14: Momentum oscillator
  - MACD: Trend follower

### Prediction Window
- **Context**: Last 60 trading days
- **Horizon**: Next 5 trading days
- **Output**: Full OHLCV forecast for each day

### Caching Layers
1. **Redis** (Request Cache): 1-day TTL, speeds up repeated queries
2. **Qdrant** (Semantic Cache): Semantic search for similar historical queries
3. **Model Checkpoints** (File System): Load-once, inference many times

### Monitoring Strategy
- **Drift Detection**: Compares reference vs current distributions
- **Agent Evaluation**: Quality metrics on generated reports
- **Prometheus**: System health (CPU, memory, request latency)
- **Grafana**: Visual dashboards for human oversight

---

## Quick Start: Where to Begin Reading

**If you want to understand:**

- **"How do I run this?"** → [README.md](README.md) + [run_docker.sh](run_docker.sh)
- **"What does each component do?"** → [src/config.py](src/config.py) → Look at constants
- **"How is data processed?"** → [src/data/ingestion.py](src/data/ingestion.py) → [src/data/preparation.py](src/data/preparation.py)
- **"How does the model work?"** → [src/model/definition.py](src/model/definition.py) → [src/pipelines/training_pipeline.py](src/pipelines/training_pipeline.py)
- **"How are predictions made?"** → [src/inference.py](src/inference.py) → [src/pipelines/inference_pipeline.py](src/pipelines/inference_pipeline.py)
- **"How do agents work?"** → [src/agents/graph.py](src/agents/graph.py) → [src/agents/nodes.py](src/agents/nodes.py)
- **"What APIs exist?"** → [backend/api.py](backend/api.py)
- **"How to monitor?"** → [src/monitoring/drift.py](src/monitoring/drift.py) + Grafana dashboard
- **"How to deploy?"** → [docker-compose.yml](docker-compose.yml) or [k8s/](k8s/) files

---

**This is a production-grade system combining modern ML, LLMs, and DevOps best practices. Start with understanding the config and data flow, then dive into specific components based on your interest!**
