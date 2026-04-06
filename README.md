# Costco Holiday Spend Prediction
  Forecasting member-level holiday spending using behavioral features, ensamble modeling, and SHAP     based interpretability.

# Project Overview
This project tackles a retail business question: can we predict how much a Costco member will spend during the holiday season based on their behavior?

Using a synthetic dataset of 5,000 members, I built an end-to-end machine learning pipeline - from frature engineering and preprocessing through model training, evaluation, and interpretation. The project also includes a Quarto-written analysis and presentation slides, demonstrating the ability to communicate technical findings to a business audience. 

## Why Simulated Data?
Simulating data with realistic statistical distributions provided me with control over the underlying statistical structure, allowing for validation of a model behavior and SHAP interpretability against known relationships, while approximating realistic retail patterns without privacy limitations.
    - Because the true relationships are known, simulated data allows validation that both the model and SHAP explanations recover the intended structure, not spurious patterns. 
    
----------------------------------------------------------------------------------------------------
  # Business Question
  Q: Which members are likely to be high holiday spenders, and what behavioral signals drive that           prediction?
  
----------------------------------------------------------------------------------------------------
  # Dataset 
  * rows = 5,000 synthetic members, generated with realistic behavioral distributions.
  * 14 features across five behavioral catgories.
    
|category         | Features                                                       |
|  -------------- | -------------------------------------------------------------- | 
|Spending history | rolling_12mo_spend, rolling_24mo_spend, rolling_12mo_avg_basket| 
|Visit behavior   | rolling_12mo_visit, weekend_shopping_ratio                     |
|Department spend |  pct_spend_grocery_consumables, pct_spend_pharmacy_health, pct_spend_home_hardware, pct_spend_auto_sporting, pct_spend_general_merch,         |
|Membership       | executive_flag, member_tenure_years                            |
|Other behavior   | return_rate, promo_usage_rate                                  |
|Target Variable  | holiday_spend(USD)                                             |

  * Mean: $1,501.20 | Stnd.dev: $566.50 | Range: $171-$5,476

# Methodology
1. Exploratory Data Analysis
   * Examined feature distributions and identified right skew in the target variable *holiday_spend*
   * Applied log1p transformation to *holiday_spend* before modeling to stabilize variance and meet       linear model assumptions.
   * All predictions were inverse-transformed back to dollar scale for evaluation.
    
2. Multicollinearity Detection and Handling
   * Built a correlation matrix across all 14 features to identify multicolliniarity.
   * Identify one highly correlated pair, *rolling_12mo_spend* and *rolling_12mo_avg_basket* with r       = 0.83
   * Model-specific feature engineering:
       * Linear models (Regression & ElasticNet): Removed the variable *rolling_12mo_avg_basket*
       * XGBoost: Retained all 14 variables, as tree models are robust to multicollinearity.
3. Data Splitting
   * 80/20 train-test split (4,000 train / 1,000 test)
   * Within training set, further 80/20 split for a validation set (3,200/800)
   * Validation set used for XGBoost early stopping; final evaluation on held-out test set only.
4. Model Training
   * Linear Regression - interpretable baseline, trained on log-transformed target.
   * ElasticNet - combines L1 + L2 regularization (alpha=0.001, l1_ratio=0.5) to handle correlated        features.
   * XGBoost - gradient boosted trees with early stopping (best iteration: round 156), L1/L2              regularization, and tuned hyperparameters (max_depth=3,eta=0.03,subsample=0.8).
     
----------------------------------------------------------------------------------------------------
# Results

|Model              | R²         | MAE        | RMSE    |
| ----------------- | ---------- | -----------|-------- |
|Linear Regression  | 0.6614     | $264.66    | $333.65 |
|ElasticNet         | 0.6625     |$264.42     | $333.11 |
|XGBoost            | 0.6864     | $256.20    | $321.07 |

* XGBoost outperformed both linear models across all metrics, explaining ~68.6% of variance in holiday spend with a mean absolute error of $256.20 - roughly 17% of the average holiday spend ($1,501).
* The narrow gap between Linear Regression and ElasticNet suggests that multicollinearity was the primary data challenge, not excess features - consistent with what the correlation matrix showed.
  
----------------------------------------------------------------------------------------------------
# Feature Importance & SHAP Analysis
  * SHAP (SHapley Additive exPlanations) was used to interpret the XGBoost model's predictions at both the global and individual level.
    
## Top Predictors (Mean Absolute SHAP Value):
1. rolling_12mo_spend - dominant predictors by a wide margin (gain=9.07); high annual spenders consistently predict high holiday spend.
2. rolling_12mo_avg_basket - average transaction size adds independent signal beyond total spend.
3. executive_flag - executive members spend significantly more, consistent with their higher engagement profile.
4. rolling_24mo_spend - longer-term spend history provides additional predictive signal.
5. rolling_12mo_visit - visit frequency is a secondary but meaningful signal.

   * **key business insight**: A member's spending momentum, as in how much they already spent, is       the strongest signal for holiday behavior- not demographic or department preference. This           suggests holiday campaigns should prioritize high-spend, high-frequency members regardless of       what category they shop.
     
----------------------------------------------------------------------------------------------------

# Residual Analysis
* Residual plots were generated for all three models. XGBoost showed tighter residual spread and no systematic bias patterns, confirming its superior fit. All models showed similar residual structure, suggesting the remaining ~31% of unexplained variance reflects genuine rendomness or unmeasured factors (e.g., income shocks, external economic conditions) rather than model limitations.
  
----------------------------------------------------------------------------------------------------

# Tools & Technologies
* Python: pandas, numpy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn
* R & Quarto: Technical analysis write-up and presentation slides (revealjs)
* Environment: Google Colab / Jupyter Notebook
  
----------------------------------------------------------------------------------------------------

# Repository Contents

|File                      | Description                                    |
| ------------------------ | ---------------------------------------------- | 
|costco_basket.ipynb       | Main Python notebook: EDA, preprocessing, modeling, evaluation                                                                                     |
|XGboost.py                | Standalone modeling script                     |
|Costco_Holiday_Spend.qmd  | Quarto technical analysis document             |
|Costco_Slides.qmd         | Quarto slide source                            |
|Costco_Slides.html        | Rendered presentation slides                   |
|Costco_simulated_data.csv | Synthetic dataset (5,000 members, 14 features) |
|Costco_SHAP.png           | SHAP feature importance visualization          |

----------------------------------------------------------------------------------------------------

# View the Presentation
Interactive slides available here: 👉 Costco Holiday Spend — Quarto Slides (https://rpubs.com/zabudamous/1385469)

----------------------------------------------------------------------------------------------------

# Key Skills Demonstrated
* Translating a business question into an end-to-end ML regression problem.
* Feature engineering and model-specific preprocessing pipelines.
* Multicollinearity detection and targeted remediation.
* Gradient boosted tree modeling with early stopping and regularization.
* Model interpretability using SHAP (global + individual level).
* Cross-model benchmarking with consistent evaluation methodology.
* Bridging Python ML with R-based reporting and presentation.
  
----------------------------------------------------------------------------------------------------

# What I'd Improve with More Time
* Hyperparameter tuning: GridSearchCV or Optuna sweep across XGBoost parameters.
* Additional models: LightGBM or CatBoost for comparison.
* Richer simulation: Add seasonal noise and member-level heterogeneity to the data-generating process.
* Segmentation layer: Cluster members by behavioral profile before modeling (member-group-specific models).
* Deployment sketch: Wrap the model in a simple Flask/FastAPI endpoint to simulate a production scoring pipeline.
  
----------------------------------------------------------------------------------------------------
**This project is part of my data science portfolio and reflects my interest in applied machine learning, statistical rigor, and data-driven business storytelling.**




