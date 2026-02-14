Got it 👍 — no extra explanations, no additions.
Below is a clean, formatted **README.md** using only *your content*, just arranged professionally.

---

# Research Methodology: Bot-IoT Detection via Modified CGAN

This repository contains the implementation of a two-stage methodology for network intrusion detection using the **Bot-IoT dataset**. The approach leverages a **Modified Conditional Generative Adversarial Network (CGAN)** with dual discriminators to improve synthetic data quality for minority attack classes.

---

## Stage 1: Data Pre-processing

The pre-processing pipeline converts raw network traffic captures into a clean, normalized format suitable for deep learning models.

### Key Steps

* **Merging:** Combines multi-part CSV files from the UNSW 2018 dataset.
* **Cleansing:** Drops identifiers like IP addresses and timestamps to prevent the model from "memorizing" specific network nodes.
* **Numeric Conversion:** Standardizes mixed-type columns into float/integer types.
* **Imputation:** Fills null values with column means to preserve dataset size.
* **Scaling:** Applies Min-Max Normalization to scale features between 0 and 1.

---

### Stage 1: Data Pre-processing Code

```python
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 1. Path and Headers
path = '/content/drive/MyDrive/Bot-IoT DataSet/'
all_files = glob.glob(path + "UNSW_2018_IoT_Botnet_Dataset_*.csv")
column_names = [
    'pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 
    'pkts', 'bytes', 'state', 'ltime', 'seq', 'dur', 'mean', 'stddev', 
    'sum', 'min', 'max', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 
    'srate', 'drate', 'TnBPDstIP', 'TnBPSrcIP', 'TNP_DstIP', 'TNP_SrcIP', 
    'T_DstIP_PLP', 'T_SrcIP_PLP', 'Anis_MOD', 'attack', 'category', 'subcategory'
]

# 2. Loading and Cleansing
df = pd.concat((pd.read_csv(f, names=column_names, low_memory=False) for f in all_files[:5]), ignore_index=True)
drop_cols = ['pkSeqID', 'stime', 'flgs', 'proto', 'saddr', 'daddr', 'sport', 'dport', 'state', 'subcategory', 'category']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# 3. Numeric Conversion and Imputation
for col in df.columns:
    if col != 'attack':
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())

# 4. Scaling
le = LabelEncoder()
df['attack'] = le.fit_transform(df['attack'].astype(str))
scaler = MinMaxScaler()
numerical_features = df.columns.drop('attack')
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

---

## Stage 2: Synthetic Data Generation (Modified CGAN)

To handle class imbalance, we utilize a CGAN architecture featuring **Dual Discriminators (D₁ and D₂).**

### Architecture & Training

**Generator**

* 3 Dense layers (128 → 256 → 512 units) with Batch Normalization.

**D₁ (Statistical)**

* Analyzes the raw distribution of the features to capture statistical trends.

**D₂ (Conditional)**

* Validates the class-wise similarity between the generated data and the requested label.

**Optimization**

* Adam optimizer with a learning rate of 0.0002.

**Stopping Criterion**

* Training terminates if generator loss < 0.05 or 200 epochs are reached.

---

### Stage 2: Synthetic Data Generation (Modified CGAN) Code

```python
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Parameters
latent_dim = 100
num_features = df.shape[1] - 1
batch_size = 128

# 1. Generator
def build_generator():
    z = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    combined = layers.Concatenate()([z, label])
    x = layers.Dense(128)(combined)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    out = layers.Dense(num_features, activation='sigmoid')(x)
    return Model([z, label], out)

# 2. Discriminators (D1 and D2)
def build_d1():
    feat = Input(shape=(num_features,))
    x = layers.Dense(512)(feat)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    return Model(feat, layers.Dense(1, activation='sigmoid')(x))

def build_d2():
    feat = Input(shape=(num_features,))
    label = Input(shape=(1,))
    combined = layers.Concatenate()([feat, label])
    x = layers.Dense(512)(combined)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    return Model([feat, label], layers.Dense(1, activation='sigmoid')(x))
```
