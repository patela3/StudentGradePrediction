# Student Grade Prediction Project

## Dataset

The data comes from a merged CSV (`merged.csv`) stored in Google Drive. It combines student survey responses with actual exam scores. Key columns include:

- **`What grade do you think you got?`** → renamed to `Letter Grade`
- **`Total Score`** → renamed to `numscore`
- Various survey response columns (categorical and numeric)

---

## Preprocessing

A shared preprocessing pipeline is applied across all three parts:

- **Numeric features**: median imputation → standard scaling
- **Categorical features**: most-frequent imputation → one-hot encoding
- Rows with missing or "Prefer not to say" letter grades are dropped
- The following columns are excluded as features: `Timestamp`, `Max Points`, `Letter Grade`, `Pass/Fail`, `numscore`, `Which section are you in?`, and the extra credit honesty column

---

## Part 1: Pass/Fail Classification

Frames the problem as binary classification — grades A/B/C map to **pass**, D/F map to **fail**.

**Models trained:** Logistic Regression, Random Forest, SVM
**80/20 stratified train/test split**

| Model               | Accuracy | F1 Score (pass) |
|---------------------|----------|-----------------|
| Logistic Regression | 91.7%    | 0.955           |
| Random Forest       | 88.9%    | 0.941           |
| SVM                 | 88.9%    | 0.941           |

Logistic Regression performed best. All models achieved strong results, likely because the pass/fail boundary is relatively easy to learn with the available features.

---

## Part 2: Letter Grade Classification (A–F)

Predicts the exact letter grade — a harder 5-class problem.

**Models trained:** Logistic Regression, Random Forest, SVM
**80/20 stratified train/test split**

| Model               | Accuracy | Macro Avg F1 |
|---------------------|----------|--------------|
| Logistic Regression | 41.7%    | 0.163        |
| Random Forest       | 41.7%    | 0.158        |
| SVM                 | 52.8%    | 0.138        |

Performance drops sharply compared to Part 1. All models struggle with minority grade classes (A, D, F), which have very few samples. The SVM gets the highest raw accuracy by collapsing nearly all predictions to B, but has the lowest macro F1 as a result.

---

## Part 3: Numeric Score Regression

Predicts the actual numeric exam score using a Random Forest Regressor.

**80/20 train/test split (not stratified)**

| Metric | Value  |
|--------|--------|
| MSE    | 66.26  |
| RMSE   | 8.14   |
| R²     | 0.053  |

The model explains only ~5% of the variance in scores, indicating the survey features are weak predictors of exact numeric performance. An RMSE of ~8 points means predictions are off by about 8 points on average.

---

## Dependencies

```
pandas
numpy
scikit-learn
google-colab
```

---

## How to Run

1. Upload `merged.csv` to your Google Drive root (`MyDrive/merged.csv`)
2. Open the notebook in Google Colab
3. Run all cells in order — each part builds on the shared preprocessing pipeline defined in Part 1

---

## Key Takeaways

- **Binary classification** (pass/fail) is feasible with these features (~92% accuracy)
- **Multi-class grade prediction** is significantly harder due to class imbalance and feature limitations
- **Regression** on raw scores performs poorly, suggesting survey responses alone don't capture enough signal to predict exact scores
