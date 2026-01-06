# 📊 Data Ingestion Pipeline - Complete Sequential Breakdown

## The Big Picture: Where Data Comes From and What We Do

```
┌─────────────────────────────────────────────────────────────────┐
│ GOAL: Convert raw stock market data into clean features         │
│       ready for machine learning model training                 │
└─────────────────────────────────────────────────────────────────┘

Real World Data Source (Yahoo Finance)
          ↓
          │ Raw OHLCV Data
          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: FETCH raw data from Yahoo Finance                      │
│ Step 2: CLEAN the data (handle MultiIndex, remove NaNs)        │
│ Step 3: ADD technical indicators (RSI, MACD)                   │
│ Step 4: VALIDATE data quality                                  │
│ Step 5: SAVE to Feature Store (Feast)                          │
│ Step 6: MATERIALIZE features to Redis (for fast serving)       │
└─────────────────────────────────────────────────────────────────┘
          ↓
     Clean Features
          ↓
Ready for Training Pipeline
```

---

## Real Example: Ingesting Data for "GOOG" (Google Stock)

Let's follow one request: `fetch_ohlcv("GOOG", start="2023-01-01")`

### **Timeline and Data Transformation**

```
START: User calls fetch_ohlcv("GOOG", "2023-01-01", None)
│
├─ Input Parameters:
│  ├─ ticker = "GOOG"
│  ├─ start = "2023-01-01"
│  ├─ end = None (means today's date)
│  └─ default from Config: context_len=60, pred_len=5, features=[...]
│
└─ Next Step: Fetch Data from Yahoo Finance
```

---

## STEP 1: FETCH RAW DATA FROM YAHOO FINANCE

### **Location in Code**: Lines 31-33

```python
df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
if df.empty:
    raise PipelineError(f"No data downloaded for {ticker}")
```

### **What Happens**:
- Calls Yahoo Finance API to download historical stock data
- Parameters:
  - `ticker="GOOG"`: Stock symbol
  - `start="2023-01-01"`: Earliest date (Google IPO was 2004, so this works)
  - `end=None`: Latest date (today)
  - `interval="1d"`: Daily data (1 day candles)
  - `auto_adjust=True`: Adjusts for stock splits and dividends
  - `progress=False`: No progress bar in console

### **Raw Data You Get Back** (Example):
```
                 Open        High         Low       Close     Volume
Date                                                                 
2023-01-01       88.69       89.56       87.16      88.79   39426200
2023-01-02       88.18       88.68       86.89      87.84   54850500
2023-01-03       86.58       87.99       85.73      87.35   52357600
2023-01-04       87.03       89.38       86.94      89.34   48216000
...
2024-12-31      193.50      195.23      192.50     194.19   28304900

Columns: Open, High, Low, Close, Volume
Rows: ~503 trading days (Jan 2023 - Dec 2024)
```

### **Why This Data**:
- **Open**: Stock price at market open
- **High**: Highest price during the day
- **Low**: Lowest price during the day
- **Close**: Stock price at market close (most important)
- **Volume**: Number of shares traded

These are the building blocks for everything else. Without this data, the system cannot work.

---

## STEP 2: CLEAN THE DATA (Handle MultiIndex Quirk)

### **Location in Code**: Lines 35-37

```python
# Flatten MultiIndex columns if present (yfinance > 0.2.40 behavior)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
```

### **What's the Problem**?

With newer versions of yfinance (>0.2.40), the DataFrame sometimes returns **nested column names** (MultiIndex):

```python
# BEFORE (MultiIndex - Problem):
            Price                                     
             Open       High        Low      Close     Volume
Date                                                              
2023-01-01   88.69      89.56      87.16     88.79   39426200

# This means df.columns = [("Price", "Open"), ("Price", "High"), ...]
# It's nested/hierarchical, hard to access
```

### **What This Code Does**:

Flattens the nested structure:

```python
# AFTER (Single-level - Fixed):
           Open       High        Low      Close     Volume
Date                                                             
2023-01-01 88.69      89.56      87.16     88.79   39426200

# Now df.columns = ["Open", "High", "Low", "Close", "Volume"]
# Much cleaner and easier to access
```

### **How It Works**:
- `isinstance(df.columns, pd.MultiIndex)`: Checks if columns are nested
- `df.columns.get_level_values(0)`: Gets the top level (e.g., "Open", "High")

### **Why This Matters**: 
Without this fix, accessing columns like `df["Close"]` would fail because the column is actually `("Price", "Close")`.

---

## STEP 3: RESET INDEX AND RENAME "Date" TO "date"

### **Location in Code**: Lines 39-40

```python
df = df.reset_index().rename(columns={"Date": "date"})
df = df[["date", "Open", "High", "Low", "Close", "Volume"]].dropna()
```

### **What Happens**:

**Before**:
```
                 Open        High         Low       Close     Volume
Date                                                                 
2023-01-01       88.69       89.56       87.16      88.79   39426200
```
(Date is the index, not a column)

**After**:
```
        date      Open       High        Low      Close     Volume
0   2023-01-01   88.69      89.56       87.16     88.79   39426200
1   2023-01-02   88.18      88.68       86.89     87.84   54850500
```
(Date is now a regular column, rows have numeric indices 0, 1, 2, ...)

**Then we filter to just these 6 columns** and drop any rows with NaN:
```
df[["date", "Open", "High", "Low", "Close", "Volume"]].dropna()
```

### **Why This Matters**:
- Date should be a column we can access like `df["date"]`, not hidden in the index
- Only keeping 6 columns ensures clean data
- `dropna()` removes incomplete rows

---

## STEP 4: ADD TECHNICAL INDICATORS (RSI and MACD)

### **Location in Code**: Lines 43-44

```python
df["RSI14"] = rsi(df["Close"])
df["MACD"] = macd(df["Close"])
```

This is where we transform raw prices into **signals that models can learn from**.

### **Sub-Function 1: RSI (Relative Strength Index)**

#### **Location in Code**: Lines 14-20

```python
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for a given series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)
```

#### **Real Example: Calculate RSI for GOOG**

**Input**: Close prices for GOOG
```
Close prices:  [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
```

**Step 1: Calculate daily price changes (delta)**
```python
delta = series.diff()
# delta = [NaN, +2, -1, +2, +2, -1, +2, +2, -1, +2]
#         (today - yesterday)
```

**Step 2: Separate gains and losses (14-period averages)**
```python
gain = [gains for 14 days, averaged]
# For simplicity (using actual 14-day period):
# gain = [average of positive deltas in last 14 days]
# loss = [average of absolute negative deltas in last 14 days]
```

For our 10-point example:
```
Days        Prices  Delta   Gain  Loss
Day 1       100     NaN     NaN   NaN     (need 14 days for RSI)
Day 2       102     +2      +2    0
Day 3       101     -1      0     +1
Day 4       103     +2      +2    0
...
Day 14+     ...     ...     X.XX  Y.YY    (now RSI is calculable)
```

**Step 3: Calculate RS (Relative Strength) and RSI**
```python
rs = gain / loss
rsi = 100 - 100 / (1 + rs)
```

**Example Values**:
- If gain = 1.5, loss = 0.5: rs = 3.0, rsi = 75 (strong uptrend - overbought)
- If gain = 0.5, loss = 1.5: rs = 0.33, rsi = 25 (strong downtrend - oversold)
- If gain = loss: rs = 1.0, rsi = 50 (neutral)

#### **Real DataFrame After Adding RSI**:

```
        date      Open   High    Low  Close  Volume    RSI14
0   2023-01-01   88.69  89.56  87.16  88.79  39426200   NaN
1   2023-01-02   88.18  88.68  86.89  87.84  54850500   NaN
...
14  2023-01-20   95.20  96.50  94.80  95.10  42000000   72.5  (overbought!)
15  2023-01-21   94.50  95.80  94.20  94.80  41000000   68.2
...
503 2024-12-31  193.50 195.23 192.50 194.19  28304900   65.1
```

### **Sub-Function 2: MACD (Moving Average Convergence Divergence)**

#### **Location in Code**: Lines 22-25

```python
def macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Calculate MACD for a given series."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow
```

#### **Real Example: Calculate MACD for GOOG**

**What is EMA (Exponential Moving Average)?**
- A weighted average where recent prices matter more than old prices
- `span=12`: Emphasizes last 12 days
- `span=26`: Emphasizes last 26 days

**Example**:
```
Close prices:  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, ...]

EMA12 (fast)  = Weighted average of last 12 prices, heavily favoring recent
              = ~107.5 (pulls toward current price 109)

EMA26 (slow)  = Weighted average of last 26 prices, more balanced
              = ~105.0 (slower to react)

MACD = EMA12 - EMA26 = 107.5 - 105.0 = +2.5
```

**What it means**:
- **MACD > 0**: Fast average is above slow → uptrend (bullish)
- **MACD < 0**: Fast average is below slow → downtrend (bearish)
- **MACD crossover**: When MACD crosses zero → signal to buy/sell

#### **Real DataFrame After Adding MACD**:

```
        date      Open   High    Low  Close  Volume    RSI14   MACD
0   2023-01-01   88.69  89.56  87.16  88.79  39426200   NaN    NaN
...
26  2023-02-02  95.00  96.20  94.80  95.50  40000000   58.0   +1.2
27  2023-02-03  96.00  97.50  95.80  97.00  42000000   62.0   +1.8  (uptrend!)
...
503 2024-12-31  193.50 195.23 192.50 194.19  28304900   65.1   +2.3
```

---

## STEP 5: SELECT FINAL FEATURES AND DROP NaNs

### **Location in Code**: Line 45

```python
df = df[["date"] + config.features].dropna()
```

### **What's in `config.features`?**

From `src/config.py`:
```python
features: List[str] = field(default_factory=lambda: [
    "Open", "High", "Low", "Close", "Volume", "RSI14", "MACD"
])
```

### **What This Line Does**:

Selects only these columns:
```python
["date", "Open", "High", "Low", "Close", "Volume", "RSI14", "MACD"]
```

And drops rows where ANY of these are NaN (missing):
```python
.dropna()
```

**Before dropna()**:
```
        date      Open   High    Low  Close  Volume    RSI14   MACD
0   2023-01-01   88.69  89.56  87.16  88.79  39426200   NaN    NaN   <- Has NaNs
1   2023-01-02   88.18  88.68  86.89  87.84  54850500   NaN    NaN   <- Has NaNs
...
25  2023-02-01  94.50  95.80  94.20  94.80  41000000   NaN    NaN   <- Has NaNs
26  2023-02-02  95.00  96.20  94.80  95.50  40000000   58.0   +1.2  <- All good
27  2023-02-03  96.00  97.50  95.80  97.00  42000000   62.0   +1.8  <- All good
...
```

**After dropna()** (rows 0-25 removed because RSI/MACD need 14-26 days):
```
        date      Open   High    Low  Close  Volume    RSI14   MACD
0   2023-02-02  95.00  96.20  94.80  95.50  40000000   58.0   +1.2
1   2023-02-03  96.00  97.50  95.80  97.00  42000000   62.0   +1.8
2   2023-02-04  97.50  99.00  96.80  98.50  43000000   65.0   +2.1
...
478 2024-12-31  193.50 195.23 192.50 194.19  28304900   65.1   +2.3
```

Now we have **~478 clean rows** instead of 503 (first 25 rows dropped due to NaN indicators).

---

## STEP 6: VALIDATE DATA QUALITY

### **Location in Code**: Lines 47-52

```python
# Check 1: Sufficient length
if len(df) < config.context_len + config.pred_len:
    raise PipelineError(f"Insufficient data for {ticker}: {len(df)} rows, need at least {config.context_len + config.pred_len}")

# Check 2: No NaN values
if df[config.features].isnull().any().any():
    raise PipelineError(f"NaN values found in features for {ticker}")

# Check 3: All numeric
if not df[config.features].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
    raise PipelineError(f"Non-numeric values found in features for {ticker}")

print(f"Fetched {len(df)} rows for {ticker}")
```

### **What Each Check Does**:

**Check 1: Do we have enough data?**
```python
if len(df) < config.context_len + config.pred_len:
    # config.context_len = 60 (historical window)
    # config.pred_len = 5 (forecast horizon)
    # Need minimum 65 rows
    
# For GOOG: We have 478 rows ✓ PASS (478 >= 65)
```

**Check 2: Are there any NaN (missing) values?**
```python
if df[config.features].isnull().any().any():
    # First .any() = any NaN in each column
    # Second .any() = any column has NaN
    
# For GOOG: No NaNs after dropna() ✓ PASS
```

**Check 3: Are all features numeric (not text)?**
```python
if not df[config.features].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
    # Checks each column's data type
    
# For GOOG: All are float64 (numeric) ✓ PASS
```

**Output**:
```
Fetched 478 rows for GOOG
```

---

## STEP 7: SAVE TO FEAST FEATURE STORE

### **Location in Code**: Lines 54-100

This is the **most complex part**. It saves the data to a centralized feature store.

### **Sub-Step 7.1: Prepare Data for Feast**

#### **Location in Code**: Lines 56-59

```python
feast_df = df.copy()
feast_df["ticker"] = ticker
feast_df["event_timestamp"] = pd.to_datetime(feast_df["date"])
feast_df["created_timestamp"] = datetime.now()
```

**What This Does**:

Adds metadata columns that Feast requires:

```
Before (ML features):
        date      Open   High    Low  Close  Volume    RSI14   MACD
0   2023-02-02  95.00  96.20  94.80  95.50  40000000   58.0   +1.2

After (Feast format):
        date      Open   High    Low  Close  Volume    RSI14   MACD   ticker  event_timestamp          created_timestamp
0   2023-02-02  95.00  96.20  94.80  95.50  40000000   58.0   +1.2   GOOG   2023-02-02 00:00:00   2024-01-04 10:30:45
```

**Why These Columns**?
- **ticker**: Entity key (what stock is this data for?)
- **event_timestamp**: When did this event occur (price on this date)?
- **created_timestamp**: When did we ingest this data?

Feast uses `ticker` as the **entity key** to group and serve features by stock.

### **Sub-Step 7.2: Create Directory and Set Up File Locking**

#### **Location in Code**: Lines 61-68

```python
repo_path = os.path.join(os.getcwd(), "feature_store")
data_path = os.path.join(repo_path, "data", "features.parquet")
os.makedirs(os.path.dirname(data_path), exist_ok=True)

import fcntl
lock_path = data_path + ".lock"
```

**What This Does**:

Creates directory structure:
```
stock-agent-ops/
└── feature_store/
    └── data/
        └── features.parquet    <- Where data goes
        └── features.parquet.lock <- Prevents concurrent access
```

**Why File Locking**?

Imagine 2 training jobs run simultaneously, both fetching data for different stocks:

```
Without locking (DISASTER):
Job1: Read features.parquet
Job2: Read features.parquet
Job1: Write (adds GOOG data)        <- Saves to disk
Job2: Write (adds TSLA data)        <- OVERWRITES Job1's changes!
Result: GOOG data is lost!

With locking (SAFE):
Job1: Lock features.parquet.lock
Job1: Read, Modify, Write features.parquet
Job1: Unlock
Job2: Waits for lock...
Job2: Lock features.parquet.lock
Job2: Read (includes GOOG data), Modify (adds TSLA), Write
Job2: Unlock
Result: Both GOOG and TSLA saved safely!
```

### **Sub-Step 7.3: Acquire Lock and Save Data**

#### **Location in Code**: Lines 70-85

```python
with open(lock_path, "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)  # Exclusive lock (blocks others)
    try:
        if os.path.exists(data_path):
            existing_df = pd.read_parquet(data_path)
            combined_df = pd.concat([existing_df, feast_df]).drop_duplicates(subset=["ticker", "event_timestamp"])
            combined_df.to_parquet(data_path)
        else:
            feast_df.to_parquet(data_path)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)  # Release lock
```

**Scenario 1: First time ingesting GOOG**

```
File doesn't exist yet
├─ Create features.parquet
└─ Write 478 GOOG rows
```

**Scenario 2: Re-ingesting GOOG (updating data)**

```
File already exists with old GOOG data
├─ Read existing data (e.g., 400 old GOOG rows)
├─ Append new data (e.g., 478 new GOOG rows)
├─ Deduplicate by (ticker, event_timestamp)
│  └─ Removes old GOOG rows that are now outdated
└─ Save combined, deduplicated data
```

**Example**:
```python
# Old data in file
old_df:
        date      Open   High  ...   ticker  event_timestamp
0   2023-02-02  95.00  96.20  ...   GOOG   2023-02-02
...
400 2023-12-31 190.00 191.50  ...   GOOG   2023-12-31

# New data we just fetched
new_df:
        date      Open   High  ...   ticker  event_timestamp
0   2023-02-02  95.00  96.20  ...   GOOG   2023-02-02  <- DUPLICATE
...
478 2024-12-31 193.50 195.23  ...   GOOG   2024-12-31

# After concat + deduplicate
combined_df:
        date      Open   High  ...   ticker  event_timestamp
0   2023-02-02  95.00  96.20  ...   GOOG   2023-02-02  <- KEPT (latest)
...
478 2024-12-31 193.50 195.23  ...   GOOG   2024-12-31
```

### **Sub-Step 7.4: Run Feast CLI Commands**

#### **Location in Code**: Lines 103-118

```python
print("🔄 Running Feast apply...")
subprocess.run(["feast", "apply"], cwd=repo_path, check=True, capture_output=True)

print("🔄 Running Feast materialization...")
subprocess.run(
    ["feast", "materialize-incremental", datetime.now().isoformat()], 
    cwd=repo_path, 
    check=True,
    capture_output=True
)
print("✅ Feast features materialized to Redis")
```

**What are these commands?**

**Command 1: `feast apply`**
```bash
cd feature_store
feast apply
```

Reads `feature_store.yaml` and `features.py`, registers them:
```yaml
# feature_store.yaml tells Feast:
# - Data source: features.parquet
# - Entity: ticker (what we group by)
# - Online store: Redis (fast access)
```

Output:
```
Created entity: ticker
Created feature_view: stock_features
Registered in online store (Redis)
```

**Command 2: `feast materialize-incremental <timestamp>`**
```bash
cd feature_store
feast materialize-incremental 2024-01-04T10:30:45
```

Loads features from Parquet **into Redis** for fast serving:

```
Before (Parquet - slow, disk-based):
feature_store/data/features.parquet (disk, ~10ms to read)

After (Redis - fast, in-memory):
Redis:
  "GOOG/2024-01-04": {"Open": 193.5, "High": 195.23, ...}
  "GOOG/2024-01-03": {"Open": 192.0, "High": 193.5, ...}
  ...
  (in-memory, ~1ms to read)
```

**Why Materialize?**
- **Training**: Reads from Parquet (historical)
- **Serving**: Reads from Redis (real-time, fast)

---

## STEP 8: ERROR HANDLING

### **Location in Code**: Lines 120-128

```python
except subprocess.CalledProcessError as e:
    print(f"⚠️ Feast command failed: {e.stderr.decode() if e.stderr else e}")
except Exception as e:
    print(f"⚠️ Feast ingestion failed: {e}")

return df
```

If Feast fails (e.g., Redis not running), it **logs a warning but still returns the DataFrame**. This is graceful degradation—the system works even if Feast isn't available.

---

## COMPLETE DATA FLOW VISUALIZATION

```
USER REQUEST
│
└─ fetch_ohlcv("GOOG", start="2023-01-01")
   │
   ├─ STEP 1: Download from Yahoo Finance
   │  └─ Raw Data (503 rows): Open, High, Low, Close, Volume
   │
   ├─ STEP 2: Handle MultiIndex columns
   │  └─ Flattened columns
   │
   ├─ STEP 3: Reset index, rename "Date" to "date"
   │  └─ Date is now a column
   │
   ├─ STEP 4: Add Technical Indicators
   │  ├─ RSI14 (momentum signal)
   │  └─ MACD (trend signal)
   │
   ├─ STEP 5: Select 7 features + date, drop NaNs
   │  └─ Clean Data (478 rows): date, Open, High, Low, Close, Volume, RSI14, MACD
   │
   ├─ STEP 6: Validate
   │  ├─ Check 1: 478 rows >= 65 ✓
   │  ├─ Check 2: No NaNs ✓
   │  └─ Check 3: All numeric ✓
   │
   ├─ STEP 7: Save to Feature Store
   │  ├─ Add Feast metadata (ticker, timestamps)
   │  ├─ Lock & write to features.parquet
   │  ├─ Run "feast apply" (register definitions)
   │  └─ Run "feast materialize" (load to Redis)
   │
   └─ RETURN: Clean DataFrame (478 rows, 8 columns)
      │
      └─ Ready for Data Preparation → Training → Model
```

---

## Key Takeaways

| Step | Input | Process | Output |
|------|-------|---------|--------|
| 1 | Ticker symbol | Download from Yahoo Finance | Raw OHLCV |
| 2-3 | Raw DataFrame | Clean columns & index | Standardized structure |
| 4 | Close prices | Calculate RSI & MACD | Technical indicators |
| 5 | All columns | Select 7 features, drop NaNs | Clean features |
| 6 | Features | Validate length, type, completeness | Quality assurance |
| 7 | Clean features | Save to Parquet & materialize to Redis | Centralized storage |

---

## What Gets Returned

After all steps, the function returns:

```python
DataFrame: GOOG_data
├─ 478 rows (trading days from Feb 2, 2023 to Dec 31, 2024)
├─ 8 columns: date, Open, High, Low, Close, Volume, RSI14, MACD
├─ All values: Numeric, no NaNs
└─ Ready for: Data Preparation (creating ML tensors)
```

This DataFrame is then used by the next pipeline stage (`src/data/preparation.py`) to create training data for the LSTM model.

---

## Real-World Example: Why Each Step Matters

**Scenario: Forgetting a step**

```
❌ No RSI/MACD (skip Step 4):
   Model only sees: Open, High, Low, Close, Volume
   Result: Model can't detect momentum changes
   → Poor predictions when trend reverses

❌ No validation (skip Step 6):
   Data has NaNs or wrong types
   Result: Model crashes during training
   → Pipeline failure

❌ No feature store (skip Step 7):
   Each training job downloads data separately
   Result: Inconsistent data, training-serving skew
   → Model works in training, fails in production

✅ All steps done correctly:
   Clean, consistent, enriched data
   → Better model performance, reliable production system
```

---

## How to Trace Data for Debugging

If data looks wrong, check:

1. **After Step 1**: Did yfinance download correct data?
   ```python
   df = yf.download("GOOG", "2023-01-01", None)
   print(df.head())  # Check first 5 rows
   ```

2. **After Step 5**: Are indicators calculated?
   ```python
   df[["date", "Close", "RSI14", "MACD"]].tail()  # Check last 5
   ```

3. **After Step 6**: Did validation pass?
   ```python
   print(len(df), df.isnull().sum())  # Check row count and NaNs
   ```

4. **In Feature Store**: Is data saved?
   ```bash
   ls -lah feature_store/data/features.parquet
   ```

---

This is the **complete sequential data ingestion pipeline**. Each step transforms data closer to what the ML model needs.
