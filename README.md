---

## Dataset

**LLMs-Sentiment-Augmented-Bitcoin-Dataset** by Danilo Corsi & Cesare Compagnano  
Source: [Hugging Face](https://huggingface.co/datasets/danilocorsi/LLMs-Sentiment-Augmented-Bitcoin-Dataset)

| Property | Value |
|---|---|
| Time Range | Feb 2018 – Jun 2024 |
| Total Rows | 2,321 daily observations |
| Missing Values | 0 |
| Split | 60 / 20 / 20 (chronological) |

**Stream 1 — Price & Technical Indicators (14 features)**  
OHLCV, RSI-14, Bollinger Bands (upper/middle/lower/bandwidth/%B), SMA-10, SMA-20, Bitcoin Macro Oscillator

**Stream 2 — On-Chain Metrics (13 features)**  
Block size, avg block size, tx count (total & per block), hash rate, mining difficulty, miner revenue, 
transaction fees (USD), unique addresses, tx volume (USD), BTC supply, market cap

**Stream 3 — Sentiment Scores (5 sources)**  
Fear & Greed Index (FNG), Crypto Bull & Bear Index (CBBI), CoinTelegraph (FinBERT), 
Bitcoin News (FinBERT), Reddit (FinBERT)

Sentiment scores for CoinTelegraph, Bitcoin News, and Reddit were computed using 
[FinBERT by ProsusAI](https://huggingface.co/ProsusAI/finbert) as `P(positive) − P(negative) ∈ [−1, 1]`.

---

## Results

### Model Comparison vs. ARIMA(5,1,0) Rolling Baseline

| Metric | CNN–GRU (Ours) | ARIMA(5,1,0) | Winner |
|---|---|---|---|
| RMSE | **0.0199** | 0.0895 | Ours |
| MAE | **0.0142** | 0.0540 | Ours |
| Directional Accuracy | **56.58%** | 52.56% | Ours |
| Pearson Correlation | 0.0145 | **0.0459** | ARIMA |

- **4% directional accuracy improvement** over ARIMA baseline  
- **73% reduction in MAE** compared to ARIMA  
- Model converges stably by Step 15–20 of 30

### Stream Ablation Study

| Stream Removed | ΔRMSE | ΔDir Accuracy |
|---|---|---|
| Price & Technical | +0.0131 | −1.32% |
| On-Chain Metrics | +0.0066 | −7.89% |
| Sentiment Scores | +0.0010 | +1.32% |

On-chain metrics provide the largest marginal gain in directional accuracy. 
Price dominates RMSE. Sentiment contribution is ambiguous across metrics.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Sequence Length | 7 days |
| Prediction Horizon | 1 day (next-day log return) |
| Stride | 6 days |
| GRU Hidden Dim | 64 |
| GRU Layers | 2 |
| Batch Size | 32 |
| Learning Rate | 1e-3 (Adam + ReduceLROnPlateau) |
| Dropout | 0.3 |
| Loss | MSE on log returns |
| Max Epochs | 100 (early stopping, patience=10, min=30) |

Hyperparameter search: 15 random configurations over hidden dim ∈ {32,64}, 
lr ∈ {1e-3, 5e-4, 1e-4}, dropout ∈ {0.2, 0.3}, stride ∈ {4,5,6}, GRU layers ∈ {1,2,3}. 
All runs logged via Weights & Biases.

---

## Key Design Decisions

**Overfitting:** Initial runs collapsed training loss below validation within 30 epochs. 
Fixed via Dropout(0.3) at every layer, BatchNorm before fusion, stride 1→6, and an 
early stopping guard requiring 30 minimum epochs.

**Prediction Horizon:** Original 7-day target was too noisy for the dataset size (~230 
effective training windows). Reduced to 1-day for stable gradient updates.

**Sentiment Stream:** Upgraded from single-source linear projection to a full CNN–GRU 
encoder matching Streams 1 & 2, expanded to 5 sources. CNN filters implicitly learn 
per-source importance weights through backpropagation.

---

## Computational Budget

| | |
|---|---|
| Hardware | NVIDIA Tesla T4 (Google Colab Pro) |
| Total Compute | ~10 hours (training + hyperparameter search) |
| Variants Trained | 4 model variants + 1 ARIMA baseline |
| Hyperparameter Configs | 30 combinations tested |

---

## Limitations

- Bitcoin-only — cross-crypto dynamics (ETH, SOL) not captured
- 6-year dataset may not generalize across full market regimes
- FinBERT inference on 25,718 rows (~7,000 words each) took 6+ hours on A100 — 
  a major iteration bottleneck
- ARIMA baseline is intentionally simple; does not isolate architectural contribution 
  from multi-stream feature benefit

## Future Work

- Extend to Ethereum, Solana, and other major assets for cross-crypto modeling
- Replace GRU encoders with Transformer-based temporal attention
- Incorporate real-time/intraday data for live forecasting
- Add a multi-modal linear baseline to cleanly isolate architectural vs. feature contribution

---

## Team Contributions

| Member | Contributions |
|---|---|
| Evan Larsen | Model design & implementation, training loop, evaluation pipeline, ablation study, W&B logging |
| Samik Kundu | Architecture co-design (attention/pooling), multi-task simplification, ARIMA baseline |
| Soham Vazirani | Data collection & preprocessing, three-stream pipeline, sliding window dataset class, chronological split |
| Soumil Nariani | On-chain & sentiment data collection, FinBERT scoring pipeline, stream merging |

---

## Paper

Full report available here: [Google Drive](https://drive.google.com/file/d/1SVyG3_8tQ-LwQcWhPSUhS_IBawwc-BAI/view?usp=sharing) *(working paper, Spring 2026)*
