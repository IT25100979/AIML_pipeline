# Comprehensive Machine Learning Pipeline & Support Vector Regression (SVR) Report

**Student ID:** IT25100979  
**Module:** Machine Learning / AIML  
**Project:** Ames Housing Price Prediction (Regression)  
**Dataset:** Ames Housing Dataset (`train.csv`: 1,460 instances, 81 columns; `test.csv`: 1,459 instances, 80 columns)  
**Assigned Individual Model:** Support Vector Regression (SVR)  
**Phase 1 Individual Focus:** Feature Engineering — Dimensionality Reduction (Principal Component Analysis - PCA)  

---

## Executive Summary & Key Results

| Metric / Parameter | Value / Selected Technique |
|:---|:---|
| **Best Overall Model** | **Support Vector Regression (SVR - RBF Kernel)** |
| **Best 5-Fold CV RMSLE** | **0.10826** (Rank 1 across all 6 group models) |
| **Hold-out RMSLE (20% Split)** | **0.12864** |
| **Hold-out MAE (Mean Absolute Error)** | **$13,972.41** |
| **Hold-out RMSE (Root Mean Squared Error)**| **$22,148.41** |
| **Hold-out Coefficient of Determination ($R^2$)**| **0.9360 (93.60% variance explained)** |
| **Optimal Preprocessing Scaler** | **`StandardScaler`** |
| **Optimal SVR Hyperparameters** | **`C = 3.0`, `epsilon = 0.05`, `gamma = 0.001`, `kernel = 'rbf'`** |
| **Target Transformation** | **$\ln(1 + \text{SalePrice}) = \text{log1p}(\text{SalePrice})$ (Skewness: 1.88 $\to$ 0.12)** |
| **Leakage Protection** | **All transformations encapsulated in `sklearn.pipeline.Pipeline` with `ColumnTransformer`** |

---

# PART 1: End-to-End Machine Learning Pipeline (Phase 1)

```
[Raw Data: train.csv (1460, 81)]
               │
               ▼
┌──────────────────────────────────────────────┐
│  Target Transformation: log1p(SalePrice)     │  Skewness reduced from +1.88 to +0.12
└──────────────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 1: Missing Data Handling               │  Structural NAs -> 'None' / 0; LotFrontage -> Neighborhood Median
└──────────────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 2: Categorical Encoding                │  18 Ordinal -> Integer Map; 26 Nominal -> OneHotEncoder(ignore)
└──────────────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 3: Outlier Handling                    │  Targeted removal of 2 extreme partial sales; log1p on 20 skewed cols
└──────────────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 4: Normalization & Scaling             │  StandardScaler (Z-score) selected over MinMax and RobustScaler
└──────────────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 5: Feature Selection                   │  Drop collinear (|r| > 0.8) & uninformative (|r| < 0.05) features
└──────────────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 6: Dimensionality Reduction (PCA)      │  IT25100979 Focus: 22 PCs for 90% var; used for EDA / 2D visualization
└──────────────────────┬───────────────────────┘
               │
               ▼
[Processed Clean Feature Matrix / Pipeline Ready for Phase 2 Modeling]
```

---

## 1. Problem Formulation & Target Transformation

- **Task**: Supervised regression to predict the continuous selling price (`SalePrice` in USD) of residential properties in Ames, Iowa.
- **Target Skewness**:
  - Raw `SalePrice` exhibits strong positive right skewness of **+1.88** with kurtosis driven by high-value properties.
  - Applying natural logarithm transformation $y = \ln(1 + \text{SalePrice})$:
    $$\text{Skewness}_{\text{raw}} = 1.88 \longrightarrow \text{Skewness}_{\text{log1p}} = 0.12$$
- **Core Theoretical Rationale for Log Transformation**:
  1. **Homoscedasticity & Normality**: Normalizes error distribution and stabilizes variance across low- and high-priced properties, satisfying linear and margin-based modeling assumptions.
  2. **Relative Error Penalization**: Minimizing RMSE on log scale is mathematically equivalent to minimizing **Root Mean Squared Logarithmic Error (RMSLE)** on dollar prices:
     $$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (\ln(1 + \hat{y}_i) - \ln(1 + y_i))^2}$$
     This prevents multi-million dollar luxury homes from disproportionately dominating optimization gradients over mid-range homes.
  3. **Inversion**: Final predictions are converted back to USD using $\hat{y}_{\text{dollar}} = \exp(\hat{y}_{\text{log}}) - 1 = \text{expm1}(\hat{y}_{\text{log}})$.

---

## 2. Step 1: Handling Missing Data (Missingness Taxonomy & Imputation)

Across `train.csv`, there were **7,829 missing cells across 19 columns**. The strategy distinguishes between structural missingness and true random missingness.

### A. Structural Missingness (Missing Not At Random - MNAR / Feature Absent)
In Ames housing, an `NA` typically indicates the *absence* of a physical amenity rather than missing data.
- **Verification**:
  - `PoolQC` (1,453 missing, 99.5%) $\longleftrightarrow$ `PoolArea == 0` (exactly 1,453 rows).
  - `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` (81 missing, 5.5%) $\longleftrightarrow$ `GarageArea == 0` (81 rows).
  - `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` (37–38 missing, 2.5%) $\longleftrightarrow$ `TotalBsmtSF == 0` (37 rows).
  - `FireplaceQu` (690 missing, 47.3%) $\longleftrightarrow$ `Fireplaces == 0` (690 rows).
  - `Alley`, `Fence`, `MiscFeature`, `MasVnrType`.
- **Treatment**:
  - Categorical absence $\to$ Imputed with explicit categorical token **`"None"`**.
  - Associated numeric measurements (e.g., `MasVnrArea`) $\to$ Imputed with **`0`**.

### B. True Missingness (Missing At Random / Completely At Random - MAR / MCAR)
- `LotFrontage` (259 missing, 17.7%): Street frontage varies systematically by geographic location and zoning. Imputed using the **median of the corresponding `Neighborhood`**, falling back to the global median ($69.0\text{ ft}$).
- `GarageYrBlt` (81 missing): Imputed using the training set median.
- `Electrical` (1 missing): Imputed using the mode (`"SBrkr"`).

### C. Test Set Protection & Pipeline Integration
`test.csv` contained 7,878 missing cells, including missing values in columns that had no missing values in training (e.g., `MSZoning`, `Utilities`, `Exterior1st`, `Exterior2nd`, `BsmtFinSF1`, `TotalBsmtSF`, `GarageCars`). All imputers are encapsulated within `ColumnTransformer` and fitted strictly on training folds to eliminate data leakage.

---

## 3. Step 2: Categorical Feature Encoding

The dataset contains **44 categorical variables** (43 text variables + `MSSubClass` which is converted from integer to string nominal category).

### A. Ordinal Encoding (18 Features)
Variables with intrinsic monotonic order are mapped to numeric integer ranks $[0, N-1]$ to preserve order information in a single column without expanding dimensionality:
- **Quality & Condition Scales** (`ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`):
  $$\{\text{'None'}: 0, \text{'Po'}: 1, \text{'Fa'}: 2, \text{'TA'}: 3, \text{'Gd'}: 4, \text{'Ex'}: 5\}$$
- **Basement Exposure** (`BsmtExposure`): `{'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}`
- **Basement Rating** (`BsmtFinType1`, `BsmtFinType2`): `{'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}`
- **Garage Finish** (`GarageFinish`): `{'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}`
- **Lot Shape** (`LotShape`): `{'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}`
- **Land Slope** (`LandSlope`): `{'Sev': 0, 'Mod': 1, 'Gtl': 2}`
- **Paved Driveway** (`PavedDrive`): `{'N': 0, 'P': 1, 'Y': 2}`
- **Central Air Conditioning** (`CentralAir`): `{'N': 0, 'Y': 1}`

### B. Nominal Encoding (26 Features)
Variables without intrinsic ordering (e.g., `Neighborhood`, `MSZoning`, `Foundation`, `Exterior1st`, `SaleType`) are encoded via **One-Hot Encoding**.
- In exploratory analysis, this expands features from 81 to 252 columns (197 binary indicators).
- In production pipelines, `OneHotEncoder(handle_unknown='ignore')` is used to gracefully ignore unseen test categories.

---

## 4. Step 3: Outlier Handling & Non-Linear Transformations

### A. Comparison of Outlier Strategies

```
Raw Training Set: 1460 Rows, Mean Price: $180,921, Houses > $400k: 28
      │
      ├─────────────────────────────────────────────┐
      ▼                                             ▼
[Candidate A: Sequential IQR Filter]         [Candidate B: Targeted Anomaly Removal]
- Checked 5 size features with IQR bounds    - Dropped ONLY 2 extreme partial sales:
- DROPPED 209 rows (14.3% of data)             (GrLivArea > 4000 sq ft & SalePrice < $300k)
- Mean Price: dropped to $173,398            - DROPPED 2 rows (0.14% of data)
- Houses > $400k: dropped from 28 to 3!       - Mean Price: preserved at $180,933
- RESULT: REJECTED (severe bias on luxury)   - Houses > $400k: preserved all 28
                                             - RESULT: CHOSEN
```

- **Why Candidate A (IQR Filtering) was Rejected**:
  - Filtering on $Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$ across `LotArea`, `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, and `LotFrontage` removed 209 homes.
  - It disproportionately truncated genuine luxury homes (reducing homes > \$400k from 28 to 3), introducing artificial survivorship bias and crippling the model's ability to predict higher-end properties present in `test.csv`.
- **Candidate B (Domain Targeted Removal - Chosen)**:
  - Focused strictly on properties with $\text{GrLivArea} > 4,000\text{ sq ft}$.
  - Kept valid luxury homes (Id 692: 4,316 sq ft sold for \$755,000; Id 1183: 4,476 sq ft sold for \$745,000).
  - Removed **only the 2 documented data anomalies** (Id 524: 4,676 sq ft sold for \$184,750; Id 1299: 5,642 sq ft sold for \$160,000 under `SaleCondition = 'Partial'`).
- **Strict Leakage Rule**: Outlier removal was applied **only to the training split** (1,166 rows kept), never to hold-out validation sets or `test.csv`.

### B. Feature Skewness Compression
Applied $\text{log1p}(x) = \ln(1 + x)$ to 20 non-negative continuous features with skewness $> 0.75$ (`LotArea`, `LotFrontage`, `MasVnrArea`, `BsmtFinSF1`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`, `GrLivArea`, `WoodDeckSF`, `OpenPorchSF`, etc.). Maximum feature skewness was reduced from 24.48 to 15.52.

---

## 5. Step 4: Normalization & Scaling

Features exist on vastly different numeric scales (`LotArea`: $1,300 - 215,245\text{ sq ft}$; `YearBuilt`: $1872 - 2010$; `OverallQual`: $1 - 10$; binary one-hot indicators: $0$ or $1$).

### Empirical Evaluation of Scalers (5-Fold CV RMSLE on KNN):
1. **`StandardScaler` ($z = \frac{x - \mu}{\sigma}$)**: **RMSLE = 0.1774 (BEST)**. Preserves zero-centered distributions with unit standard deviation.
2. **`MinMaxScaler` ($x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$)**: **RMSLE = 0.1903**. Bounded in $[0, 1]$, but compresses dense regions when outliers exist.
3. **No Scaling**: **RMSLE = 0.2217**. Distance dominated entirely by `LotArea` and square footages.
4. **`RobustScaler` ($x_{\text{scaled}} = \frac{x - Q_2}{\text{IQR}}$)**: **RMSLE = 0.2485 (WORST)**. For sparse features (e.g., `PoolArea`, `3SsnPorch`, `MiscVal`), $\text{IQR} \approx 0$, causing division by near-zero and massive variance explosion (hold-out values reaching 14.87).

---

## 6. Step 5: Feature Selection (Correlation & Multicollinearity Analysis)

### A. Target Correlation ($r$ with `LogPrice`)
- Highest positive predictors: `OverallQual` ($r = 0.817$), `GrLivArea` ($r = 0.701$), `GarageCars` ($r = 0.681$), `GarageArea` ($r = 0.651$), `TotalBsmtSF` ($r = 0.612$), `1stFlrSF` ($r = 0.597$), `FullBath` ($r = 0.595$), `YearBuilt` ($r = 0.587$).

### B. Collinear Pair Redundancy Removal ($|r| > 0.80$)
When two predictors are collinear, the one with lower target correlation is dropped:
1. `GarageCars` vs `GarageArea` ($r = 0.88$) $\to$ Dropped `GarageArea` (kept `GarageCars`, $r = 0.681 > 0.651$).
2. `GrLivArea` vs `TotRmsAbvGrd` ($r = 0.83$) $\to$ Dropped `TotRmsAbvGrd` (kept `GrLivArea`, $r = 0.701 > 0.534$).
3. `YearBuilt` vs `GarageYrBlt` ($r = 0.83$) $\to$ Dropped `GarageYrBlt` (kept `YearBuilt`, $r = 0.587 > 0.541$).
4. `TotalBsmtSF` vs `1stFlrSF` ($r = 0.82$) $\to$ Dropped `1stFlrSF` (kept `TotalBsmtSF`, $r = 0.612 > 0.597$).

### C. Low-Relevance Filtering ($|r| < 0.05$)
Dropped uninformative numeric features: `OverallCond` ($-0.037$), `BsmtFinSF2` ($-0.005$), `LowQualFinSF` ($-0.038$), `BsmtHalfBath` ($-0.005$), `MiscVal` ($-0.020$), `YrSold` ($-0.037$).

### D. Trade-off Analysis & Final Modeling Decision
- Ridge CV RMSLE: **0.1607 (all 35 features)** vs **0.1672 (selected 25 features)**.
- *Viva Defense Insight*: Linear correlation discards non-linear predictive relationships (e.g., `OverallCond` has a strong non-linear U-shaped pricing effect). Consequently, Phase 2 models keep all features and let regularized models / kernels handle sparsity and weight shrinkage.

---

## 7. Step 6: Dimensionality Reduction via PCA (IT25100979 Phase 1 Focus)

### A. Mathematical Formulation
Principal Component Analysis finds orthogonal axes $w_1, w_2, \dots, w_k$ that maximize variance of standardized input matrix $X$:
$$\max_{\|w\|=1} \text{Var}(Xw) = \max_{\|w\|=1} w^T \Sigma w$$
where $\Sigma = \frac{1}{n} X^T X$ is the empirical covariance matrix. Principal components are eigenvectors of $\Sigma$ ordered by decreasing eigenvalues $\lambda_i$.

### B. Variance Analysis on Ames Housing (35 Numeric Features)
- **PC1 & PC2**: Explain **29.4%** of total variance (PC1 = 17.5%, PC2 = 11.9%).
- **Variance Thresholds**:
  - **90% Variance**: Requires **22 components** (out of 35).
  - **95% Variance**: Requires **25 components**.
- **PC Loadings**:
  - PC1 is dominated by overall size and quality: `OverallQual` (+0.302), `GrLivArea` (+0.300), `GarageCars` (+0.283), `GarageArea` (+0.279), `FullBath` (+0.263), `TotalBsmtSF` (+0.258).
  - PC2 captures house morphology and vertical distribution: `TotRmsAbvGrd` (+0.341), `GrLivArea` (+0.263), `TotalBsmtSF` (-0.190), `1stFlrSF` (-0.126).

```
PCA Explained Variance Scree Data:
PC1: 17.5% | PC2: 11.9% | Cumulative (PC1+PC2): 29.4%
Components for 90% Variance: 22 / 35
Components for 95% Variance: 25 / 35
```

### C. Component vs. Error Experiment (Ridge 5-Fold CV RMSLE)

| Components ($k$) | 5-Fold CV RMSLE |
|:---:|:---:|
| 2 | 0.1821 |
| 5 | 0.1719 |
| 10 | 0.1710 |
| 15 | 0.1694 |
| 20 | 0.1643 |
| **35 (All Features)** | **0.1607 (Best)** |

### D. Critical Viva Examination Takeaways for PCA:
1. **Error Decreases Monotonically**: Predictive error falls steadily as more components are included. PCA does *not* improve regression accuracy for Ames Housing; it acts solely as lossy compression.
2. **Unsupervised Limitation**: PCA maximizes total feature variance without reference to the target $y$ (`SalePrice`). Subtle low-variance components contain crucial pricing signals.
3. **Loss of Interpretability**: Principal components are linear combinations of all 35 features, making it impossible to interpret individual feature coefficients (e.g., dollar value per square foot).
4. **Final Decision**: PCA is retained for exploratory 2D visualization (revealing a clear price gradient along PC1), but is intentionally omitted from the final regression modeling in Phase 2 in favor of kernel SVR and regularized regression.

---

# PART 2: In-Depth Breakdown of Support Vector Regression (SVR - IT25100979 Phase 2)

---

## 1. Core Theory: How Support Vector Regression Works

Support Vector Regression (SVR) is the continuous extension of Support Vector Machines formulated by Vladimir Vapnik.

```
       y (SalePrice_log)
       ▲
       │                  /  f(x) + ε  (Upper Margin Boundary)
       │          ●      /
       │                /  ● (Support Vector with slack ξ_i > 0)
       │   ────────────/───────────────────────────────
       │              /  f(x) = <w, φ(x)> + b (Regression Function)
       │   ──────────/─────────────────────────────────
       │            /   ● (Support Vector with slack ξ_i* > 0)
       │           /  f(x) - ε  (Lower Margin Boundary)
       │          /
       └─────────/──────────────────────────────────────► x (Standardized Features)
                 │◄── 2ε ──►│ (ε-Insensitive Loss Tube: Zero Penalty Inside)
```

### A. The $\epsilon$-Insensitive Loss Function ($L_\epsilon$)
Unlike Ordinary Least Squares (OLS) which penalizes all squared residuals, SVR uses an $\epsilon$-insensitive loss function:
$$L_\epsilon(y, f(x)) = \begin{cases} 0 & \text{if } |y - f(x)| \le \epsilon \\ |y - f(x)| - \epsilon & \text{otherwise} \end{cases}$$
- Any prediction falling within a tube of radius $\epsilon$ around the true price receives **zero penalty**.
- Points outside the tube receive a linear penalty proportional to their distance from the margin.

### B. Primal Optimization Formulation
The goal is to find a flat function $f(x) = \langle w, \phi(x) \rangle + b$ that minimizes prediction error while penalizing model complexity:
$$\min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)$$
Subject to:
$$\begin{cases} y_i - (\langle w, \phi(x_i) \rangle + b) \le \epsilon + \xi_i \\ (\langle w, \phi(x_i) \rangle + b) - y_i \le \epsilon + \xi_i^* \\ \xi_i, \xi_i^* \ge 0 \end{cases}$$
where:
- $\frac{1}{2} \|w\|^2$: Regularization term enforcing maximum flatness and margin maximization.
- $C$: Regularization parameter balancing margin flatness against penalty for training errors exceeding $\epsilon$.
- $\xi_i, \xi_i^*$: Slack variables measuring errors above and below the $\epsilon$-tube.

### C. Dual Formulation & Support Vectors
Using Lagrange multipliers ($\alpha_i, \alpha_i^* \ge 0$), the dual optimization problem becomes:
$$\max_{\alpha, \alpha^*} -\frac{1}{2} \sum_{i,j=1}^n (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*) K(x_i, x_j) - \epsilon \sum_{i=1}^n (\alpha_i + \alpha_i^*) + \sum_{i=1}^n y_i (\alpha_i - \alpha_i^*)$$
Subject to $\sum_{i=1}^n (\alpha_i - \alpha_i^*) = 0$ and $0 \le \alpha_i, \alpha_i^* \le C$.

The final prediction function is:
$$f(x) = \sum_{i \in \text{SV}} (\alpha_i - \alpha_i^*) K(x_i, x) + b$$
- **Sparsity**: Only data points on or outside the $\epsilon$-tube have non-zero Lagrange multipliers ($\alpha_i - \alpha_i^* \neq 0$). These points are the **Support Vectors**. All points inside the tube do not affect the model.

### D. Radial Basis Function (RBF) Kernel
The RBF kernel projects input features into an infinite-dimensional Hilbert space:
$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$
where $\gamma = \frac{1}{2\sigma^2}$ governs the kernel bandwidth (radius of influence of each support vector).

---

## 2. Features Used in SVR Training

All **79 raw input features** were utilized via pipeline transformations:

1. **35 Numeric Features**:
   `LotFrontage`, `LotArea`, `OverallQual`, `OverallCond`, `YearBuilt`, `YearRemodAdd`, `MasVnrArea`, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `LowQualFinSF`, `GrLivArea`, `BsmtFullBath`, `BsmtHalfBath`, `FullBath`, `HalfBath`, `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`, `Fireplaces`, `GarageYrBlt`, `GarageCars`, `GarageArea`, `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `3SsnPorch`, `ScreenPorch`, `PoolArea`, `MiscVal`, `MoSold`, `YrSold`.
   - *Pipeline Transformation*: Imputed with `SimpleImputer(strategy='median')` $\to$ Scaled with `StandardScaler()`.

2. **44 Categorical Features**:
   `MSSubClass` (converted to string), `MSZoning`, `Street`, `Alley`, `LotShape`, `LandContour`, `Utilities`, `LotConfig`, `LandSlope`, `Neighborhood`, `Condition1`, `Condition2`, `BldgType`, `HouseStyle`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`, `MasVnrType`, `ExterQual`, `ExterCond`, `Foundation`, `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `Heating`, `HeatingQC`, `CentralAir`, `Electrical`, `KitchenQual`, `Functional`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `PavedDrive`, `PoolQC`, `Fence`, `MiscFeature`, `SaleType`, `SaleCondition`.
   - *Pipeline Transformation*: Imputed with `SimpleImputer(strategy='constant', fill_value='None')` $\to$ One-Hot Encoded with `OneHotEncoder(handle_unknown='ignore')`.

3. **Total Feature Space**:
   After one-hot encoding categorical variables across training folds, the input feature space expands to **280+ dense/sparse columns**.

---

## 3. Data Preprocessing & Pipeline Architecture

```
                                    Raw Training Split (X_train)
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
         [35 Numeric Features]                                       [44 Categorical Features]
                   │                                                           │
                   ▼                                                           ▼
  SimpleImputer(strategy='median')                         SimpleImputer(fill_value='None')
                   │                                                           │
                   ▼                                                           ▼
         StandardScaler()                                   OneHotEncoder(handle_unknown='ignore')
                   │                                                           │
                   └─────────────────────────────┬─────────────────────────────┘
                                                 │
                                                 ▼
                                     ColumnTransformer (Union)
                                                 │
                                                 ▼
                                      SVR(kernel='rbf', C=3,
                                          epsilon=0.05, gamma=0.001)
                                                 │
                                                 ▼
                                      Predicted log(SalePrice)
                                                 │
                                                 ▼
                                       expm1 -> Predicted USD ($)
```

- **Split Setup**: 80% Training ($1,168$ samples) vs 20% Hold-out ($292$ samples) with `random_state=42`.
- **Targeted Training Outlier Removal**: 2 extreme partial sales dropped from training only $\to$ **1,166 training instances**. Hold-out split ($292$ instances) remained completely untouched.
- **Leakage Prevention**: All imputation medians, one-hot category mappings, and scaling means/standard deviations were computed strictly on the training folds during cross-validation.

---

## 4. Hyperparameter Tuning & Cross-Validation Process

- **Cross-Validation Scheme**: 5-Fold Cross-Validation (`KFold(n_splits=5, shuffle=True, random_state=42)`).
- **Optimization Criterion**: Negative Root Mean Squared Error on log prices (`scoring='neg_root_mean_squared_error'`).
- **Hyperparameter Grid (`param_grid`)**:
  - `preprocess__numeric__scale`: `[StandardScaler(), MinMaxScaler()]` (2 options)
  - `regressor__C`: `[0.3, 1.0, 3.0, 10.0]` (4 options)
  - `regressor__epsilon`: `[0.01, 0.05]` (2 options)
  - `regressor__gamma`: `['scale', 0.001]` (2 options)
  - **Total Evaluations**: $2 \times 4 \times 2 \times 2 = 32$ parameter configurations $\times 5$ folds = **160 trained models**.

### Optimal Hyperparameters Selected by Grid Search:

| Hyperparameter | Optimal Value | Parameter Role & Viva Explanation |
|:---|:---:|:---|
| **Numeric Scaler** | **`StandardScaler()`** | Standardizes numeric inputs to $\mu=0, \sigma=1$. Outperformed `MinMaxScaler` across all combinations. |
| **Regularization $C$** | **`3.0`** | Controls penalty for margin violations. $C=3.0$ prevents underfitting while regularizing against high-leverage outliers. |
| **Insensitivity $\epsilon$** | **`0.05`** | Errors within $\approx 5\%$ on log price incur zero loss, creating a sparse support vector representation and filtering noise. |
| **Kernel Bandwidth $\gamma$** | **`0.001`** | Small gamma enforces smooth, wide Gaussian influence, preventing localized overfitting common with default `gamma='scale'`. |

---

## 5. Model Evaluation Metrics & Quantitative Results

### A. Validation & Hold-Out Performance (Evaluated ONCE on 20% Split)

| Evaluation Stage / Metric | Value | Interpretation |
|:---|:---:|:---|
| **5-Fold Cross-Validation RMSLE** | **0.10826** | Average relative log error across training folds |
| **Hold-out RMSLE** | **0.12864** | Generalization error on 292 unseen houses |
| **Hold-out RMSE** | **$22,148.41** | Dollar error penalizing larger errors |
| **Hold-out MAE** | **$13,972.41** | Average absolute prediction error |
| **Hold-out $R^2$ Score** | **0.9360** | **93.60% of price variance explained** |

### B. Diagnostic Interpretation
1. **Generalization Gap**: The difference between 5-fold CV RMSLE ($0.1083$) and hold-out RMSLE ($0.1286$) is small ($\Delta = 0.0203$), indicating robust generalization without overfitting.
2. **Residual Distribution**: Residuals ($y_{\text{actual}} - \hat{y}_{\text{pred}}$) are symmetrically distributed around zero with near-zero mean, confirming absence of systematic under- or over-pricing bias.
3. **Unlabeled Test Predictions (`test.csv`: 1,459 houses)**: Full pipeline refit on all 1,458 training samples yielded predictions ranging from **$55,561 to $548,664**, accurately matching Ames market distributions.

---

# PART 3: Six-Model Group Comparison — Why SVR is the Best Model

---

## 1. Group Model Comparison Table

All six members trained individual models using identical 80/20 data splits, 5-fold cross-validation, and hold-out evaluation:

| Rank | Model | Member ID / Notebook | Scaler | 5-Fold CV RMSLE | Hold-out RMSLE | Hold-out RMSE ($) | Hold-out MAE ($) | Hold-out $R^2$ | Optimal Hyperparameters |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **Support Vector Regression (SVR)** | **IT25100979 / MEMBER4** | **StandardScaler** | **0.10826** | **0.12864** | **$22,148** | **$13,972** | **0.9360** | `C=3, epsilon=0.05, gamma=0.001` |
| 2 | Ridge Regression | IT25103946 / MEMBER1 | StandardScaler | 0.11106 | 0.12954 | $22,014 | $14,764 | 0.9368 | `alpha=30` |
| 3 | Hist Gradient Boosting | IT25102867 / MEMBER6 | StandardScaler | 0.11964 | 0.13186 | $24,674 | $15,102 | 0.9206 | `lr=0.1, max_depth=3, max_iter=300` |
| 4 | Random Forest | IT25101882 / MEMBER3 | StandardScaler | 0.13741 | 0.14491 | $28,438 | $16,966 | 0.8946 | `n_estimators=200, min_samples_leaf=1` |
| 5 | K-Nearest Neighbors | IT25101923 / MEMBER5 | StandardScaler | 0.16194 | 0.18196 | $38,843 | $21,648 | 0.8033 | `n_neighbors=8, weights='distance'` |
| 6 | Decision Tree | IT25103725 / MEMBER2 | MinMaxScaler | 0.18774 | 0.19121 | $37,785 | $24,287 | 0.8139 | `max_depth=12, min_samples_leaf=10` |

---

## 2. Core Theory: Why SVR Outperformed All Other Algorithms

```
Performance Ranking (Hold-out RMSLE - Lower is Better):
SVR (0.1286) < Ridge (0.1295) < HistGBM (0.1319) < RandomForest (0.1449) < KNN (0.1820) < DecisionTree (0.1912)
```

### 1. Robustness in High Dimensions ($d \approx 280 > n = 1,166$)
After one-hot encoding, the dataset expands to over 280 features for 1,166 training instances. Tree-based algorithms (Decision Trees, Random Forests) struggle with fragmented sample splits along sparse dummy columns. In contrast, SVR's capacity is determined by the margin ($\|w\|$) and support vector density, making it fundamentally immune to the curse of dimensionality.

### 2. Tolerance to Housing Price Noise ($\epsilon$-Insensitivity)
Real estate transactions contain negotiation variance and appraisal noise ($\approx \pm 3-5\%$). OLS, Ridge, and Tree models attempt to fit this noise, increasing variance. SVR's $\epsilon = 0.05$ creates a dead zone where errors under 5% receive zero loss, focusing model capacity strictly on true structural drivers.

### 3. Smooth Non-Linear Boundary (RBF Kernel vs. Tree Discretization)
Housing prices follow smooth non-linear relationships (e.g., diminishing marginal returns on square footage and interactive quality multipliers). Decision Trees and Random Forests approximate smooth surfaces using axis-aligned orthogonal step functions, requiring deep trees that overfit. SVR with an RBF kernel constructs smooth, continuous non-linear manifolds.

### 4. Resistance to Outliers via Linear Slack Penalties
Squared error methods (OLS, Ridge) penalize large residuals quadratically ($\xi^2$), causing extreme prices to disproportionately distort regression coefficients. SVR penalizes violations linearly ($C\xi$), providing intrinsic robustness against anomalous houses.

### 5. Lowest Mean Absolute Error ($13,972)
SVR achieved the lowest MAE of any model — beating Ridge by **$792**, HistGradientBoosting by **$1,130**, and Random Forest by **$2,994** — demonstrating superior real-world pricing accuracy.

---

# PART 4: Viva Exam Defense — Key Questions & Precise Technical Answers

### Q1: Why did you apply a logarithmic transformation to `SalePrice`?
> **Answer**: `SalePrice` has a raw positive skewness of +1.88 with high-value luxury homes extending the right tail. Log transformation ($\text{log1p}$) reduces skewness to +0.12, stabilizing residual variance (homoscedasticity) and fulfilling normality assumptions for linear/margin models. Furthermore, optimizing RMSE on the log scale is mathematically equivalent to minimizing Root Mean Squared Logarithmic Error (RMSLE), which optimizes relative percentage error so that errors on \$100k homes are penalized proportionally to errors on \$800k homes.

### Q2: Why did you reject the sequential IQR outlier filter in Phase 1?
> **Answer**: Applying standard IQR filtering ($Q_1 - 1.5\text{IQR}$ to $Q_3 + 1.5\text{IQR}$) across five size features removed 209 homes (14.3% of the dataset) and collapsed homes over \$400k from 28 down to 3. This introduced severe survivorship bias by stripping out genuine high-end homes, dragging mean price down from \$180.9k to \$173.4k and crippling test set prediction on luxury homes. Instead, we performed targeted removal of only 2 documented data anomalies ($\text{GrLivArea} > 4000\text{ sq ft}$ with $\text{SalePrice} < \$300\text{k}$) from the training set only.

### Q3: Why is `StandardScaler` superior to `RobustScaler` and `MinMaxScaler` for SVR in this project?
> **Answer**: SVR uses an RBF kernel based on Euclidean distance: $\|x_i - x_j\|^2$. `StandardScaler` centers continuous features at $\mu=0$ and scales variance to $\sigma=1$, giving balanced weight across features. `RobustScaler` divides by IQR; for sparse features with mostly zeros (e.g., `PoolArea`, `MiscVal`, `3SsnPorch`), $\text{IQR} \approx 0$, causing division by near-zero and massive variance explosion (hold-out values exceeding 14.8). `MinMaxScaler` compresses dense feature distributions when outliers exist. Empirical 5-fold CV confirmed `StandardScaler` achieved the lowest error across all models.

### Q4: What was your Phase 1 individual contribution (PCA), and why wasn't PCA used in the final SVR model?
> **Answer**: Under IT25100979, I conducted the Dimensionality Reduction analysis using PCA. PC1 and PC2 explained only 29.4% of total variance, and 22 components were needed to retain 90% variance. In CV experiments with Ridge, error decreased monotonically as components were added (RMSLE 0.1821 with 2 PCs $\to$ 0.1607 with all 35 features). Because PCA is an unsupervised technique that maximizes total variance rather than correlation with `SalePrice`, low-variance components still contain critical pricing information. Furthermore, PCA destroys feature interpretability. Thus, PCA was retained for 2D visualization but excluded from the final regression pipelines.

### Q5: Explain the physical role of hyperparameters $C$, $\epsilon$, and $\gamma$ in your tuned SVR model.
> **Answer**: 
> - **$C = 3.0$**: Controls the trade-off between maximizing margin flatness ($\frac{1}{2}\|w\|^2$) and penalizing training errors outside the $\epsilon$-tube. A moderate $C=3.0$ prevents overfitting while ensuring sufficient model flexibility.
> - **$\epsilon = 0.05$**: Sets the radius of the insensitivity tube ($\approx 5\%$ relative error on log scale). Errors within this tube incur zero penalty, inducing sparsity in support vectors and filtering valuation noise.
> - **$\gamma = 0.001$**: Defines the RBF kernel bandwidth ($\gamma = \frac{1}{2\sigma^2}$). A small $\gamma=0.001$ ensures a smooth, wide Gaussian influence across support vectors, preventing localized over-fitting that occurs with default `gamma='scale'`.

### Q6: How did your pipeline design guarantee zero data leakage?
> **Answer**: We partitioned data into an 80% training set and 20% hold-out set before any preprocessing. All missing value imputers (`SimpleImputer`), categorical encoders (`OneHotEncoder`), and numeric scalers (`StandardScaler`) were encapsulated within an `sklearn.compose.ColumnTransformer` inside an `sklearn.pipeline.Pipeline`. During 5-fold cross-validation, transformations were fitted strictly on the 4 training folds and applied to the validation fold. The 20% hold-out set was evaluated only once at the end.

### Q7: Why did SVR outperform tree ensembles like Random Forest and Histogram Gradient Boosting?
> **Answer**: SVR with an RBF kernel constructs smooth, continuous non-linear decision manifolds, whereas tree ensembles approximate functions using axis-aligned orthogonal step functions. On a dataset of moderate sample size ($n=1,166$) with over 280 one-hot encoded dimensions, decision trees suffer from sparse split fragmentation. SVR regularizes via margin maximization in Hilbert space and ignores minor appraisal noise via $\epsilon$-insensitivity, resulting in lower variance, superior hold-out RMSLE ($0.1286$), and the lowest MAE ($\$13,972$).

---
*Report generated and validated against empirical project outputs under Student ID IT25100979.*
