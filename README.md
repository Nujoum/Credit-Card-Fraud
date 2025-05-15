# Credit Card Fraud Detection Pipeline

## 1. Project Overview
Developed a production-ready fraud-screening system for a U.S. government credit-card ledger of 98,393 transactions (Jan 1 – Dec 31, 2010). The objective was to automatically intercept the riskiest transactions—minimizing fraud losses while preserving legitimate spending.

## 2. Data Description
- **Volume & Period:** 98,393 records spanning one calendar year.  
- **Fields:**  
  - Card identifier, merchant category, transaction type  
  - Timestamp, transaction amount  
  - Merchant state, ZIP code  
  - Binary fraud label (ground truth)  
- **Class Imbalance:** 2.53 % fraud overall; rises to 40.45 % for transactions > \$5,000.

## 3. Data Cleaning & Preprocessing
1. **Malformed Data Handling**  
   - Standardized merchant state codes; imputed missing ZIP codes via k-NN on geographic centroids.  
2. **Outlier Treatment**  
   - Winsorized transaction amounts at the 99.5th percentile.  
3. **Timestamp Features**  
   - Extracted hour-of-day, day-of-week, week-of-year, seasonality flags.  
4. **Geospatial Features**  
   - Calculated Haversine distance between consecutive transactions.  
5. **Behavioral Aggregates**  
   - Rolling windows over 1 hr, 24 hr, 7 day: transaction counts, sum of amounts, average interval.

## 4. Feature Engineering & Selection
- **Initial Pool:** ~4,400 raw and derived metrics (velocity, ratios, flags, distances).  
- **Univariate Filtering:** KS-test to retain top 20 % most discriminatory features.  
- **Wrapper Selection:** Forward and backward stepwise search using LightGBM and CatBoost importance; final feature set of 20 variables.

## 5. Model Development & Evaluation
| Model                 | AUC (Test) | Precision @ 3 % Review | Recall @ 3 % Review | FDR (3 % Review) |
|-----------------------|------------|------------------------|---------------------|------------------|
| Decision Tree         | 0.88       | 0.21                   | 0.66                | 0.79             |
| Random Forest         | 0.95       | 0.28                   | 0.72                | 0.75             |
| LightGBM (GOSS)       | **0.97**   | **0.35**               | **0.78**            | **0.70**         |
| LightGBM + SMOTE      | 0.96       | 0.30                   | 0.75                | 0.73             |
| Neural Network (MLP)  | 0.94       | 0.26                   | 0.70                | 0.77             |
| CatBoost              | 0.96       | 0.32                   | 0.76                | 0.72             |

- **Winner:** LightGBM with Gradient-based One-Side Sampling (GOSS) for best tradeoff between detection power and false positive control.  
- **Hyperparameters (final):**  
  - `num_leaves=128`, `max_depth=10`, `learning_rate=0.05`  
  - `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`  
  - Early stopping at 50 rounds on validation AUC.

## 6. Financial Impact Analysis
- **Savings Estimation:**  
  - Annual fraud exposure ≈\$70 M.  
  - Blocking top 8 % of risk scores yields \~\$56.7 M net savings (99 % of maximum) at 8 % review rate.  
  - False positives cost (customer service, investigation) capped at \$4 M/year under this cutoff.  
- **Recommendation:** Deploy threshold corresponding to 8 % highest-risk score to maximize net benefit before FP cost cliff.

