# Group GROUP_ID - IT2011 AI and Machine Learning

Year 2 Semester 1, 2026 - Faculty of Computing

> **Before submitting:** replace the placeholders `MEMBER1`...`MEMBER6` (file names, folder names and notebook text) with your real IT numbers, and `GROUP_ID` with your group id. A project-wide find-and-replace is enough; the member id in `results/individual_models/<id>/` and in `group_comparison/Compare_Six_Models.ipynb` must match the renamed folders.

## Project overview

This project uses the **Ames Housing** dataset to predict the **sale price of a house** (a regression task). `data/raw/train.csv` has 1,460 houses, 79 input features (36 numeric-looking, of which `MSSubClass` is really a category, and 43 text categorical) plus `Id` and the target `SalePrice`. `data/raw/test.csv` has 1,459 houses with the same features but **no `SalePrice`**, so it cannot be scored locally. Because of that, each model is evaluated on a **20% hold-out split of `train.csv`** (random_state 42), and `test.csv` is used only to produce price predictions at the end.

The team explores missing values, categorical encoding, skewed/outlier values, scaling, feature selection and PCA. The individual model notebooks use their own imputation + `OneHotEncoder` + numeric-scaling pipelines, fitted inside cross-validation. `group_pipeline.ipynb` creates an exploratory processed feature table; the six model notebooks train directly from the raw `train.csv`.

## Member contributions

| Member | Preprocessing contribution | Notebook |
|---|---|---|
| MEMBER1 | Missing-data analysis: structural `NA` vs genuine gaps, neighbourhood-median fill | `notebooks/MEMBER1_Handling_Missing_Data.ipynb` |
| MEMBER2 | Ordinal mapping of quality scales + one-hot encoding of nominal variables | `notebooks/MEMBER2_Encoding_Categorical_Variables.ipynb` |
| MEMBER3 | Outlier impact analysis, targeted removal and log transforms | `notebooks/MEMBER3_Outlier_Handling.ipynb` |
| MEMBER4 | Standard / Min-Max / Robust scaling and their effect on KNN | `notebooks/MEMBER4_Normalization_Scaling.ipynb` |
| MEMBER5 | Feature selection by correlation (redundancy + relevance) | `notebooks/MEMBER5_Feature_Engineering_Feature_Selection.ipynb` |
| MEMBER6 | PCA dimensionality reduction and visualization | `notebooks/MEMBER6_Feature_Engineering_DimensionalityReduction.ipynb` |

Each member also has an individual model notebook:

| Member | Model | Notebook |
|---|---|---|
| MEMBER1 | Ridge Regression | `individual_models/MEMBER1_Ridge_Regression/Ridge_Regression.ipynb` |
| MEMBER2 | Decision Tree | `individual_models/MEMBER2_Decision_Tree/Decision_Tree.ipynb` |
| MEMBER3 | Random Forest | `individual_models/MEMBER3_Random_Forest/Random_Forest.ipynb` |
| MEMBER4 | Support Vector Regression (RBF) | `individual_models/MEMBER4_Support_Vector_Regression/Support_Vector_Regression.ipynb` |
| MEMBER5 | K-Nearest Neighbors | `individual_models/MEMBER5_K_Nearest_Neighbors/K_Nearest_Neighbors.ipynb` |
| MEMBER6 | Histogram Gradient Boosting | `individual_models/MEMBER6_Histogram_Gradient_Boosting/Histogram_Gradient_Boosting.ipynb` |

## Results (20% hold-out split)

| Model | Hold-out RMSLE | RMSE ($) | MAE ($) | R² |
|---|---:|---:|---:|---:|
| Support Vector Regression | **0.1286** | 22,148 | **13,972** | 0.936 |
| Ridge Regression | 0.1295 | **22,014** | 14,764 | **0.937** |
| Histogram Gradient Boosting | 0.1319 | 24,674 | 15,102 | 0.921 |
| Random Forest | 0.1446 | 27,983 | 16,924 | 0.898 |
| K-Nearest Neighbors | 0.1820 | 38,843 | 21,648 | 0.803 |
| Decision Tree | 0.1912 | 37,785 | 24,287 | 0.814 |

The top three models are within about 0.003 RMSLE of each other, which is smaller than the noise expected from a 292-house hold-out set; treat them as a tie. See `VIVA_NOTES.md`.

## Outlier handling decision

A sequential IQR row filter on five size columns would remove 209 of 1,460 houses (14.3%), mostly large, expensive ones: the mean price would fall from $180.9k to $173.4k and houses above $400k from 28 to 3. That distorts the price range the model must predict, so it is rejected. The project instead removes only the **two documented anomalies** (`GrLivArea` > 4,000 sq ft with `SalePrice` < $300k; Ids 524 and 1299), models the **log of the price**, and applies `log1p` to strongly skewed numeric features in the exploratory pipeline. In the model notebooks the two rows are dropped from the **training split only**.

## Run the project

Install the dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Run the preprocessing notebooks from `notebooks/` so their `../data/raw/` paths resolve. Run `group_pipeline.ipynb` from the project root. Each individual model notebook searches the current folder and its parents for the project data; it saves its hold-out metrics, tuning CV table, actual-vs-predicted plot, metric plots and `test.csv` predictions under `results/individual_models/<member id>/`.

After all six individual model result files exist, run `group_comparison/Compare_Six_Models.ipynb`. It creates `results/six_model_comparison/six_model_comparison.csv` and `six_model_rmsle.png`, and writes the best model's price predictions to `results/outputs/submission_best_model.csv` (columns `Id`, `SalePrice`).

The saved outputs of every notebook (tables and graphs) are already included in the notebooks themselves, so you can open them without re-running anything. Re-running everything takes several minutes (the Support Vector and Gradient Boosting grid searches are the slowest). `VIVA_NOTES.md` summarises the validation approach, results and talking points.

## Optional dashboard

From the project root, run:

```bash
streamlit run app.py
```

## Local website demo

To view the project as a website without installing extra packages, double-click `start_local_website.bat` from the project folder (needs Python on PATH). It starts a local-only server and opens the dashboard at `http://127.0.0.1:8000/website/`. The dashboard shows the saved model metrics, actual-vs-predicted plots and EDA images. Close the minimized server window to stop the local server.

The dashboard displays saved comparison metrics; it is not a live price predictor.
