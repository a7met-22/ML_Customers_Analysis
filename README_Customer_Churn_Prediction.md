# E-commerce Customer Churn Prediction: Preprocessing, Model Comparison, and Tuning

An end-to-end **machine learning workflow** that predicts whether an
e-commerce customer will churn. It prepares the data, engineers
behavioral features, compares several classifiers with default
settings, tunes the strongest candidates with cross-validated search,
selects a champion model, interprets feature importance, and simulates
predictions for new customers.

The notebook demonstrates a staged approach to a classification
project, from data handling to model selection and business
interpretation, and records its conclusions next to the code that
produced them.

> **Project:** ML Customers Behavior (Churn Prediction)\
> **Notebook:** `ML_Customers_behavior_Project.ipynb`\
> **Domain:** Machine Learning · Classification · Feature Engineering ·
> Hyperparameter Tuning · Scikit-learn · XGBoost · Python\
> **Environment:** Google Colab

------------------------------------------------------------------------

## 1. Project Overview

The goal is to predict the binary target `Churned` from customer
profile, behavior, and value features, and to identify which behaviors
drive churn so that retention efforts can be targeted.

The workflow is organized in six stages:

1.  **Data handling:** load the dataset, fill missing values, encode
    categorical features, and engineer new features.
2.  **Baseline training:** train several models with default
    parameters, scaling the features for models that need it.
3.  **Hyperparameter tuning:** tune Random Forest, Gradient Boosting,
    and XGBoost with `GridSearchCV`.
4.  **Evaluation and selection:** compare the tuned models and select a
    champion.
5.  **Interpretation:** analyze precision and recall, plot the
    confusion matrix, and rank feature importance.
6.  **Simulation:** generate random new customers and predict their
    churn probability with the champion model.

The central engineering principle is:

> **A model is only as trustworthy as the process behind it: prepare
> the data carefully, compare models fairly, and report results with
> the evidence that supports them.**

------------------------------------------------------------------------

## 2. Key Features

### Data preparation

-   Fills missing numeric values with the column median and missing
    categorical values with the mode.
-   Encodes categorical features such as gender, country, and city
    into integers with `LabelEncoder`.
-   Splits the data into 80% training and 20% test sets with a fixed
    random seed.

### Feature engineering

-   **Support Intensity:** customer service calls relative to
    membership length.
-   **High-Value Abandonment:** lifetime value multiplied by cart
    abandonment rate.
-   **Purchase Efficiency:** total purchases relative to login
    frequency.
-   **Total Engagement:** email open rate plus social media engagement
    score.
-   Brings the feature set to 28 columns (24 original plus 4
    engineered).

### Model comparison

-   Trains multiple classifiers with default parameters, including
    Logistic Regression, SVM, KNN, Naive Bayes, Random Forest, Gradient
    Boosting, and XGBoost.
-   Applies `StandardScaler` for the scale-sensitive models and uses
    raw features for tree-based models.
-   Compares accuracy, precision, recall, and F1-score, sorted by F1.

### Hyperparameter tuning

-   Runs `GridSearchCV` with 3-fold cross-validation and F1 scoring for
    Random Forest, Gradient Boosting, and XGBoost.
-   Extracts the best parameters and re-evaluates each tuned model on
    the test set.
-   Includes a separate `RandomizedSearchCV` experiment for Random
    Forest.

### Interpretation and diagnostics

-   Confusion matrix for the tuned Gradient Boosting model.
-   Feature importance ranking from Random Forest, plotted as a bar
    chart of churn drivers.
-   5-fold cross-validation accuracy for stability checking.

### New-customer simulation

-   Generates five random customers, applies the same feature
    engineering, orders the columns like the training data, and
    predicts churn and churn probability.

------------------------------------------------------------------------

## 3. Architecture

| Stage | Responsibility |
|-------|----------------|
| Data loading | Reads the churn CSV into Pandas and checks types and missing values. |
| Imputation | Median for numeric columns, mode for categorical columns. |
| Encoding | Converts categorical columns to integer codes. |
| Feature engineering | Adds four derived behavioral features. |
| Split and scaling | Train/test split and `StandardScaler` for scale-sensitive models. |
| Baseline comparison | Trains default models and reports four classification metrics. |
| Tuning | Grid search with cross-validation for the three strongest models. |
| Champion selection | Compares tuned models and selects the best by F1-score. |
| Interpretation | Confusion matrix, feature importance, and cross-validation. |
| Simulation | Predicts churn for synthetic new customers. |

### Metrics used

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall share of correct predictions. |
| Precision | Of customers flagged as churners, how many truly churned. |
| Recall | Of all actual churners, how many were detected. |
| F1-score | Balance between precision and recall; used for model selection. |
| Cross-validation accuracy | Stability check across data subsets. |

### High-level workflow

``` text
          Customer churn CSV
                  |
                  v
    Impute missing values (median / mode)
                  |
                  v
      Encode categorical features
                  |
                  v
   Engineer 4 behavioral features (28 total)
                  |
                  v
     Split 80/20 + scale for LR, SVM, KNN, NB
                  |
                  v
     Baseline comparison (default parameters)
                  |
                  v
   GridSearchCV: RF, Gradient Boosting, XGBoost
                  |
                  v
   Evaluate tuned models, select champion (F1)
                  |
                  v
   Confusion matrix + feature importance + CV
                  |
                  v
   Simulate predictions for new customers
```

------------------------------------------------------------------------

## 4. Technical Highlights

### Staged, documented workflow

Each stage has a heading and a written purpose, and conclusions sit next
to the code. This makes the reasoning easy to follow from raw data to
the final model choice.

### Domain-driven features

The engineered features express business ideas: how much support a
customer needs relative to tenure, how valuable a customer is when they
abandon carts, and whether logins turn into purchases. These give the
models ratios and interactions that raw columns do not show directly.

### Scaling only where needed

Scale-sensitive models (Logistic Regression, SVM, KNN, Naive Bayes)
receive standardized features, while tree-based models use the original
values, which avoids unnecessary preprocessing for models that do not
need it.

### Tuning with an appropriate metric

Grid search uses F1 as its scoring metric, which balances precision and
recall and is more informative than accuracy when churners are the
minority class.

### Reported results

The notebook's written conclusions report the following for the tuned
models (test set):

| Model | Accuracy | Notes |
|-------|----------|-------|
| Gradient Boosting (champion) | 91.93% | F1-score 0.8471, precision 92.82%, recall 77.91%. |
| XGBoost | 91.86% | Very close runner-up. |
| Random Forest | 91.65% | Stable performance. |

A mean 5-fold cross-validation accuracy of 91.51% is reported for
stability. The top churn drivers named in the summary are customer
service calls, lifetime value, and cart abandonment rate.

------------------------------------------------------------------------

## 5. Technology Stack

| Technology | Role |
|------------|------|
| Python | Core implementation language. |
| Google Colab | Notebook execution environment. |
| Pandas / NumPy | Data handling and numeric operations. |
| Scikit-learn | Splitting, scaling, encoding, models, search, and metrics. |
| XGBoost | Gradient-boosted tree classifier. |
| Matplotlib / Seaborn | Confusion matrix and feature importance plots. |

------------------------------------------------------------------------

## 6. Running the Notebook

### Requirements

The notebook is designed for Google Colab and expects the dataset file
`ecommerce_customer_churn_dataset.csv` in the Colab working directory.
Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, and Seaborn are
needed.

Before running:

-   Open the notebook in Google Colab.
-   Upload the CSV to `/content/` or change the path in the loading
    cell.
-   Make sure the dataset includes the target `Churned` and the feature
    columns used in the feature-engineering and simulation cells.

### Execution order

The intended stages are:

1.  Load the data and check missing values.
2.  Impute missing values and encode categorical features.
3.  Engineer the four new features.
4.  Split into train and test sets, then scale.
5.  Define the model set and run the baseline comparison.
6.  Define the parameter grids and run the grid search.
7.  Evaluate the tuned models and plot the confusion matrix.
8.  Run the feature importance, cross-validation, and randomized-search
    cells.
9.  Run the new-customer simulation.

Important: the notebook as saved performs the train/test split **before**
imputation, encoding, and feature engineering, and it references objects
that are not defined in it (the model dictionary and the parameter
grids). To run it, move the split after feature engineering and define
those objects first.

### Example output

The baseline cell prints a table of accuracy, precision, recall, and
F1-score per model. The simulation cell prints, for five random
customers, a churn prediction and a churn probability. Exact results
depend on the dataset and the random seed.

------------------------------------------------------------------------

## 7. Configuration

Settings are defined inline so they can be reviewed and adjusted.

Review these settings:

-   Dataset path and the target column name (`Churned`).
-   Train/test split ratio (0.2) and `random_state` (42).
-   Imputation strategy (median and mode).
-   Feature-engineering formulas and the `+ 1` smoothing terms.
-   Models in the comparison and which receive scaled features.
-   Parameter grids, cross-validation folds (3), and scoring (`f1`).
-   Randomized search settings (`n_iter=5`).
-   Cross-validation folds for the stability check (5).

Do not commit private customer data to a public repository.

------------------------------------------------------------------------

## 8. Limitations and Data Considerations

This notebook is an educational modeling project, not a production
churn system.

| Area | Current approach | Important limitation |
|------|------------------|----------------------|
| Saved results | Code and written conclusions are in the notebook. | The saved copy has no stored outputs (the loading cell ended with a missing-file error), so the reported metrics cannot be verified from the file alone. |
| Cell order | Stages are described in order. | The split happens before imputation, encoding, and feature engineering, and the markdown says to update the split but no cell does; the model dictionary and parameter grids are used but never defined. |
| Data leakage | Imputation, encoding, and engineering run on the full dataset. | Statistics such as medians and modes should be learned from the training set only, ideally inside a pipeline. |
| Categorical encoding | `LabelEncoder` on gender, country, and city. | Integer codes impose an artificial order on nominal data, which hurts distance-based and linear models, and the encoder is refit for each column so mappings are lost. |
| Model selection | Champion chosen by F1 on the test set. | The test set is used both to compare and to pick the final model, which gives an optimistic estimate; a separate validation set or nested cross-validation is safer. |
| Stability claim | Mean cross-validation accuracy of 91.51%. | The cross-validation cell evaluates the Random Forest, not the Gradient Boosting champion, and a similar score does not prove the model is not overfitting. |
| Regression metrics | MAE, MSE, RMSE, and R² are computed. | These are regression metrics applied to a classification task; precision, recall, F1, and ROC-AUC are more appropriate. |
| Simulation | Five random customers with hand-chosen ranges. | The ranges are not derived from the data, the values are unseeded, and random inputs say nothing about real-world performance. |
| Deployment claim | The conclusion recommends production deployment. | No model is saved, no pipeline is packaged, and no monitoring or drift checks exist. |
| Feature importance | Random Forest impurity importance. | It can favor high-cardinality or correlated features and does not show causal drivers. |
| Data cleaning | Missing values are filled. | Data-quality issues such as impossible ages found during exploration are not handled in this notebook. |

### Recommendations before production use

-   Restructure the notebook so every step runs top to bottom, and save
    the outputs.
-   Split first, then fit imputers, encoders, and scalers on the
    training data only, using a scikit-learn `Pipeline`.
-   Use one-hot or target encoding for nominal categories.
-   Select models with cross-validation or a validation set, and use
    the test set once for the final estimate.
-   Report ROC-AUC, PR-AUC, and the class balance alongside F1.
-   Evaluate the champion model itself with cross-validation.
-   Replace random simulation with held-out real records.
-   Save the trained pipeline and add monitoring before any deployment.

------------------------------------------------------------------------

## 9. Repository Structure

``` text
.
├── ML_Customers_behavior_Project.ipynb
└── README.md
```

The dataset file is not included and should be added separately, only if
it contains no private customer data.

------------------------------------------------------------------------

## 10. Possible Future Improvements

-   Build a single preprocessing and modeling `Pipeline` to prevent
    leakage.
-   Apply the data-cleaning decisions from the exploration stage, such
    as handling impossible ages.
-   Compare class-imbalance strategies (class weights, resampling).
-   Add ROC-AUC and PR-AUC, and tune the decision threshold for the
    business cost of missed churners.
-   Add calibration checks for the predicted probabilities.
-   Use SHAP or permutation importance for more reliable
    interpretation.
-   Validate with a time-based split if the data has a time dimension.
-   Package the model and expose it through a simple API or app.
-   Add monitoring for data drift and performance decay.

------------------------------------------------------------------------

## 11. Key Takeaway

This project demonstrates a complete supervised learning workflow for
churn prediction: preparing data, engineering meaningful features,
comparing models fairly, tuning the strongest ones, and interpreting the
result in business terms.

The most important design lesson is that **a strong metric is not the
end of the work: trustworthy results need leakage-free preprocessing,
honest evaluation on untouched data, and conclusions that are backed by
the evidence the notebook actually produces.**
