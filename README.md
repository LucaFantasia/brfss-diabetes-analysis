# CDC BRFSS Diabetes Analysis

An exploratory, statistical and machine learning analysis of health, lifestyle and demographic indicators associated with self-reported prediabetes or diabetes in the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset.

The project combines exploratory data analysis, statistical hypothesis testing and classical machine learning models, with particular attention given to the challenges of class-imbalanced classification.

> This project is intended for statistical and machine learning analysis only. It is not a clinical diagnostic or screening tool.

## Live Dashboard

**Streamlit App:** https://brfssdiabetesanalysis.streamlit.app/

The interactive dashboard presents the main exploratory findings, statistical analysis and machine learning results from the project.

## Dataset

The project uses a cleaned version of the **2015 CDC Diabetes Health Indicators** dataset published by Alex Teboul on Kaggle.
`diabetes_binary_health_indicators_BRFSS2015.csv` is the full imbalanced binary dataset used in this project, containing:

- 253,680 observations
- 21 predictor variables
- No missing values
- Health, lifestyle and demographic indicators
- Binary target: `Diabetes_binary`
  - `0` — no diabetes
  - `1` — prediabetes or diabetes

Dataset:  
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data

## Project Structure

```text
brfss-diabetes-analysis/
│
├── app/
│   ├── streamlit_app.py
│   ├── utils.py
│   └── pages/
│       ├── overview.py
│       ├── exploratory_analysis.py
│       ├── statistical_analysis.py
│       └── model_performance.py
│
├── data/
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
│
├── notebooks/
│   └── diabetes_analysis.ipynb
│
├── outputs/
│   ├── cv_results.csv
│   ├── validation_results.csv
│   ├── test_results.csv
│   └── best_params.json
│
├── src/
│   ├── config.py
│   ├── evaluation.py
│   └── train_models.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Analysis

### Exploratory Data Analysis

The exploratory data analysis examines how the prevalence of the positive prediabetes/diabetes class varies across health, lifestyle and demographic indicators.

Several notable patterns were observed:

- Prevalence increases substantially as self-reported general health worsens.
- Older age groups generally show higher prevalence.
- High blood pressure, high cholesterol, difficulty walking and cardiovascular history are associated with notably higher prevalence.
- BMI shows a clear relationship with the target, with prevalence increasing across higher BMI ranges.
- Poor physical health shows a stronger relationship than poor mental-health days.
- Higher income and education categories are associated with lower observed prevalence.

These are observational relationships and should not be interpreted as causal effects.

### Statistical Analysis

Different statistical methods were applied according to feature type:

- **Binary features:** Chi-square test of independence and Cramér's V
- **Ordinal features:** Spearman rank correlation
- **Quantitative features:** Mann–Whitney U test and rank-biserial correlation

Benjamini-Hochberg correction was used to account for multiple comparisons.

Because the dataset contains more than 250,000 observations, very small p-values are common. Effect sizes were therefore considered alongside statistical significance.

## Machine Learning

The following classifiers were compared:

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Class-weighted versions of the main classifiers were also evaluated to investigate the effect of the imbalanced target.

### Evaluation Strategy

The data was split using stratified sampling:

- 70% training
- 10% validation
- 20% test

Five-fold stratified cross-validation was performed on the training set.

Hyperparameter tuning was performed for the Decision Tree, Random Forest and XGBoost models, with 
**PR-AUC and balanced accuracy used as the primary and secondary model-selection metric respectively**.

The test set was kept untouched until final model selection had been completed.

## Model Results

Weighted XGBoost achieved the strongest overall discrimination.

| Metric | Final Test Result |
| --- | ---: |
| Accuracy | 0.723 |
| Balanced Accuracy | **0.751** |
| Precision | 0.307 |
| Recall | 0.789 |
| F1 Score | 0.442 |
| ROC-AUC | **0.831** |
| PR-AUC | **0.438** |

Five-fold cross-validation produced approximately:

- ROC-AUC: **0.831 ± 0.002**
- PR-AUC: **0.436 ± 0.004**

The relatively small cross-validation variation suggests that performance was stable across the training folds.

### Effect of Class Weighting

Class weighting produced one of the clearest findings of the modelling analysis.

Weighted models substantially increased recall for the minority class, but this came at the cost of lower precision. For XGBoost, weighting changed the operating point considerably while ROC-AUC and PR-AUC remained almost unchanged.

This suggests that weighting primarily affected the precision-recall trade-off at the default classification threshold rather than substantially improving the model's underlying ranking ability.

## Key Findings

The project highlights several useful observations:

1. Health indicators contain meaningful predictive information about prediabetes/diabetes status.
2. More complex models improved discrimination only modestly over Logistic Regression.
3. Accuracy is misleading for this dataset because of the substantial class imbalance.
4. Class weighting can greatly improve minority-class recall, but at a significant precision cost.
5. Effect sizes are particularly important when interpreting statistical tests on very large datasets.

## Limitations

- The dataset is based largely on self-reported survey information.
- The analysis identifies associations rather than causal relationships.
- Many statistical tests are univariate and do not account for interactions or confounding between predictors.
- The positive class combines respondents with prediabetes and diabetes.
- The target is substantially imbalanced.
- The data represents an older BRFSS survey and may not fully reflect current population health patterns.
- The final classifier trades lower precision for substantially higher minority-class recall and should not be used as a clinical screening or diagnostic system.

## Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/brfss-diabetes-analysis.git
cd brfss-diabetes-analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Re-run Model Training

```bash
python src/train_models.py
```

The training pipeline saves cross-validation, validation and final test results to the `outputs/` directory.

## Technologies

- Python
- pandas
- NumPy
- SciPy
- statsmodels
- scikit-learn
- XGBoost
- Plotly
- Streamlit
- Matplotlib

## License

The source code in this repository is released under the MIT License.

The dataset is not covered by the repository's MIT License. Please refer to the original dataset source for its licensing and acknowledgement requirements.
