"""
Train and evaluate classical ML models on the CDC BRFSS diabetes dataset.

The script compares Logistic Regression, Decision Tree, Random Forest and
XGBoost classifiers, including class-weighted variants for the imbalanced
diabetes target.

A stratified 70/10/20 train/validation/test split is used. Hyperparameter
selection is performed using 5-fold stratified cross-validation on the training
set, with PR-AUC used as the primary selection metric.

The validation set is used to compare fitted models. The model with near highest
validation PR-AUC and balanced accuracy is evaluated once on an untouched test set.

Outputs:
    outputs/cv_results.csv
    outputs/validation_results.csv
    outputs/test_results.csv
    outputs/best_params.json
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from config import (
    ALL_FEATURES,
    CV_FOLDS,
    CV_SCORING,
    DATA_PATH,
    FOREST_PARAM_DISTRIBUTIONS,
    FOREST_SEARCH_ITERATIONS,
    METRIC_LABELS,
    NOMINAL_FEATURES,
    OUTPUT_DIR,
    SEED,
    SCALED_FEATURES,
    TARGET,
    TEST_SIZE,
    TREE_PARAM_GRID,
    VALIDATION_SIZE,
    XGBOOST_PARAM_DISTRIBUTIONS,
    XGBOOST_SEARCH_ITERATIONS
)

from evaluation import (
    evaluate_classifier,
    extract_cross_validate_results,
    extract_search_results,
    get_positive_class_probability,
    save_json,
    save_results
)


def load_data():
    """
    Load the BRFSS dataset and separate predictors from the target.
    """

    df = pd.read_csv(DATA_PATH)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


def split_data(X, y):
    """
    Create a stratified 70/10/20 train, validation and test split.
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SIZE, stratify=y_train_val, random_state=SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_linear_preprocessor():
    """
    Standardise ordinal and quantitative features for Logistic Regression while
    leaving binary variables in their original 0/1 representation.
    """

    return ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), SCALED_FEATURES),
            ("binary", "passthrough", NOMINAL_FEATURES)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def build_cross_validator():
    """
    Create the stratified cross-validation strategy used during training.
    """

    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def train_simple_models(X_train, y_train, cross_validator):
    """
    Cross-validate and fit the Dummy baseline and Logistic Regression models.
    """

    models = {
        "Dummy Classifier": Pipeline([
            ("model", DummyClassifier(strategy="most_frequent"))
        ]),

        "Logistic Regression": Pipeline([
            ("preprocessor", build_linear_preprocessor()),
            ("model", LogisticRegression(
                solver="lbfgs",
                max_iter=1000
            ))
        ]),

        "Logistic Regression — balanced": Pipeline([
            ("preprocessor", build_linear_preprocessor()),
            ("model", LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced"
            ))
        ])
    }

    fitted_models = {}
    cv_rows = []

    for model_name, model in models.items():
        print(f"\nCross-validating {model_name}...")
        cv_results = cross_validate(model, X_train, y_train, scoring=CV_SCORING, cv=cross_validator, n_jobs=-1)
        cv_rows.append(extract_cross_validate_results(model_name, cv_results,METRIC_LABELS))
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

    return fitted_models, cv_rows


def train_decision_trees(X_train, y_train, cross_validator):
    """
    Tune standard and class-balanced Decision Tree classifiers using grid search.
    """

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Decision Tree — balanced": DecisionTreeClassifier(class_weight="balanced", random_state=SEED)
    }

    fitted_models = {}
    cv_rows = []
    best_params = {}

    for model_name, estimator in models.items():
        print(f"\nTuning {model_name}...")

        pipeline = Pipeline([("model", estimator)])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=TREE_PARAM_GRID,
            scoring=CV_SCORING,
            refit="pr_auc",
            cv=cross_validator,
            n_jobs=-1,
            return_train_score=False
        )

        search.fit(X_train, y_train)

        fitted_models[model_name] = search.best_estimator_
        best_params[model_name] = search.best_params_

        cv_rows.append(extract_search_results(model_name, search, METRIC_LABELS))

        print(f"Best PR-AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    return fitted_models, cv_rows, best_params


def train_random_forests(X_train, y_train, cross_validator):
    """
    Tune standard and class-balanced Random Forest classifiers using randomised
    hyperparameter search.
    """

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
        "Random Forest — balanced": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=SEED, n_jobs=1
        )
    }

    fitted_models = {}
    cv_rows = []
    best_params = {}

    for model_name, estimator in models.items():
        print(f"\nTuning {model_name}...")

        pipeline = Pipeline([
            ("model", estimator)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=FOREST_PARAM_DISTRIBUTIONS,
            n_iter=FOREST_SEARCH_ITERATIONS,
            scoring=CV_SCORING,
            refit="pr_auc",
            cv=cross_validator,
            random_state=SEED,
            n_jobs=-1,
            return_train_score=False
        )

        search.fit(X_train, y_train)

        fitted_models[model_name] = search.best_estimator_
        best_params[model_name] = search.best_params_

        cv_rows.append(extract_search_results(model_name, search, METRIC_LABELS))

        print(f"Best PR-AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    return fitted_models, cv_rows, best_params


def train_xgboost_models(X_train, y_train, cross_validator):
    """
    Tune standard and class-weighted XGBoost classifiers using randomised search.

    The weighted model uses the ratio of negative to positive training examples
    as scale_pos_weight.
    """

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(f"\nXGBoost scale_pos_weight: {scale_pos_weight:.3f}")

    models = {
        "XGBoost": XGBClassifier(
            objective="binary:logistic", tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=1
        ),
        "XGBoost — weighted": XGBClassifier(
            objective="binary:logistic", tree_method="hist", eval_metric="logloss", 
            scale_pos_weight=scale_pos_weight, random_state=SEED, n_jobs=1
        )
    }

    fitted_models = {}
    cv_rows = []
    best_params = {}

    for model_name, estimator in models.items():
        print(f"\nTuning {model_name}...")

        pipeline = Pipeline([
            ("model", estimator)
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
            n_iter=XGBOOST_SEARCH_ITERATIONS,
            scoring=CV_SCORING,
            refit="pr_auc",
            cv=cross_validator,
            random_state=SEED,
            n_jobs=-1,
            return_train_score=False
        )

        search.fit(X_train, y_train)

        fitted_models[model_name] = search.best_estimator_
        best_params[model_name] = search.best_params_

        cv_rows.append(extract_search_results(model_name, search, METRIC_LABELS))

        print(f"Best PR-AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    best_params["scale_pos_weight"] = scale_pos_weight

    return fitted_models, cv_rows, best_params


def evaluate_models(fitted_models, X, y):
    """
    Evaluate all fitted models using the project's common classification metrics.
    """

    results = []

    for model_name, model in fitted_models.items():
        predictions = model.predict(X)
        probabilities = get_positive_class_probability(model, X)

        results.append(evaluate_classifier(model_name, y, predictions, probabilities))

    return pd.DataFrame(results)


def select_final_model(fitted_models, validation_results):
    """
    Select the final model based on validation PR-AUC and balanced accuracy.
    """

    best_pr_auc = validation_results["PR-AUC"].max()
    candidate_models = validation_results[
        validation_results["PR-AUC"] >= best_pr_auc - 0.010
    ]
    selected_model_name = candidate_models.loc[candidate_models["Balanced accuracy"].idxmax(), "Model"]
    model = fitted_models[selected_model_name]

    return selected_model_name, model


def evaluate_final_model(model_name, model, X_test, y_test):
    """
    Evaluate the selected model once on the untouched test set.
    """

    predictions = model.predict(X_test)
    probabilities = get_positive_class_probability(model, X_test)

    result = evaluate_classifier(model_name, y_test, predictions, probabilities)

    return pd.DataFrame([result])


def main():
    """
    Run the complete training, validation, model selection and test pipeline.
    """

    print("Loading BRFSS diabetes dataset...")

    X, y = load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"\nTraining rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows:       {len(X_test):,}")

    print("\nDiabetes prevalence:")
    print(f"Train:      {y_train.mean():.2%}")
    print(f"Validation: {y_val.mean():.2%}")
    print(f"Test:       {y_test.mean():.2%}")

    cross_validator = build_cross_validator()

    all_models = {}
    all_cv_rows = []
    all_best_params = {}

    # Dummy and Logistic Regression
    simple_models, simple_cv_rows = train_simple_models(X_train, y_train, cross_validator)

    all_models.update(simple_models)
    all_cv_rows.extend(simple_cv_rows)

    # Decision Trees
    tree_models, tree_cv_rows, tree_params = train_decision_trees(X_train, y_train, cross_validator)

    all_models.update(tree_models)
    all_cv_rows.extend(tree_cv_rows)
    all_best_params.update(tree_params)

    # Random Forests
    forest_models, forest_cv_rows, forest_params = train_random_forests(X_train, y_train, cross_validator)

    all_models.update(forest_models)
    all_cv_rows.extend(forest_cv_rows)
    all_best_params.update(forest_params)

    # XGBoost
    xgboost_models, xgboost_cv_rows, xgboost_params = train_xgboost_models(X_train, y_train, cross_validator)

    all_models.update(xgboost_models)
    all_cv_rows.extend(xgboost_cv_rows)
    all_best_params.update(xgboost_params)

    # Cross-validation results
    cv_results = pd.DataFrame(all_cv_rows)
    cv_results = cv_results.sort_values("PR-AUC mean", ascending=False).reset_index(drop=True)

    # Validation results
    validation_results = evaluate_models(all_models, X_val, y_val)

    validation_results = validation_results.sort_values("PR-AUC", ascending=False).reset_index(drop=True)

    # Final model selection
    selected_model_name, selected_model = select_final_model(all_models, validation_results)

    selected_pr_auc = validation_results.loc[validation_results["Model"] == selected_model_name, "PR-AUC"].iloc[0]

    print(f"\nSelected final model: {selected_model_name}")
    print(f"Validation PR-AUC: {selected_pr_auc:.4f}")

    # Final test evaluation
    test_results = evaluate_final_model(selected_model_name, selected_model, X_test, y_test)

    # Save experiment metadata
    all_best_params["selected_model"] = selected_model_name
    all_best_params["selection_reason"] = "Near-best PR-AUC with higher balanced accuracy"
    all_best_params["random_state"] = SEED
    all_best_params["cv_folds"] = CV_FOLDS

    # Save results
    save_results(cv_results, OUTPUT_DIR / "cv_results.csv")
    save_results(validation_results, OUTPUT_DIR / "validation_results.csv")
    save_results(test_results, OUTPUT_DIR / "test_results.csv")
    save_json(all_best_params, OUTPUT_DIR / "best_params.json")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFinal test performance:")
    print(test_results.to_string(index=False))


if __name__ == "__main__":
    main()