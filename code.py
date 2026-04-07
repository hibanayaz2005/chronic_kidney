"""
Chronic Kidney Disease (CKD) Detection
Based on: "Improving Chronic Kidney Disease Detection Efficiency:
Fine Tuned CatBoost and Nature-Inspired Algorithms with Explainable AI"

Dataset: kidney_disease.csv  (400 rows × 26 cols, UCI CKD schema)
──────────────────────────────────────────────────────────────────
Pipeline
  1. Data Loading & Cleaning (dirty labels, whitespace)
  2. Preprocessing
       a. Label Encoding (categorical → numeric)
       b. KNN Imputation (numeric) + Mode Imputation (categorical)
       c. Cuckoo Search – Outlier Adjustment  (Algorithm 1)
       d. One-Way ANOVA – Feature Significance
       e. StandardScaler
       f. Simulated Annealing – Feature Selection (Algorithm 2)
  3. Train/Test Split  80 : 20
  4. SMOTE  – Data Balancing
  5. Model Training with Grid Search + 5-Fold CV
       Logistic Regression | MLP | Random Forest | CatBoost (proposed)
  6. Evaluation
       Accuracy · AUC · Precision · Recall · F1 · Cohen's Kappa
       Confusion Matrix · ROC Curve · Learning Curve
  7. Explainability – SHAP (CatBoost)
  8. All figures saved to /mnt/user-data/outputs/
"""

# ─────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────
import warnings, math, time, random
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.impute             import KNNImputer
from sklearn.model_selection    import (train_test_split, StratifiedKFold,cross_val_score, GridSearchCV,learning_curve)
from sklearn.linear_model       import LogisticRegression
from sklearn.neural_network     import MLPClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (accuracy_score, precision_score,recall_score, f1_score,roc_auc_score, confusion_matrix,cohen_kappa_score, roc_curve,classification_report)
from sklearn.feature_selection  import f_classif

from imblearn.over_sampling import SMOTE
from catboost               import CatBoostClassifier
import shap

# ─────────────────────────────────────────────────────────────────
# 1.  DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────
DATA_PATH   = "kidney_disease.csv"
OUTPUT_DIR  = "."

# Columns exactly as they appear in kidney_disease.csv
NUMERICAL_COLS   = ["age", "bp", "sg", "al", "su",
                     "bgr", "bu", "sc", "sod", "pot",
                     "hemo", "pcv", "wc", "rc"]
CATEGORICAL_COLS = ["rbc", "pc", "pcc", "ba",
                    "htn", "dm", "cad", "appet", "pe", "ane"]
TARGET_COL       = "classification"


def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load CSV, drop id column, strip whitespace/tab artifacts from
    string columns, and normalise the target to 0/1.
    """
    df = pd.read_csv(path, na_values='?')

    # Drop the id column – not a feature
    df.drop(columns=["id"], inplace=True, errors="ignore")

    # Strip leading/trailing whitespace & embedded tabs from every
    # string column (dm has '\tno', '\t yes', etc.; target has 'ckd\t')
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Fix multi-valued string noise  (e.g. " yes" → "yes")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    # Normalise target → 1 = CKD, 0 = not CKD
    df[TARGET_COL] = df[TARGET_COL].str.lower().str.strip()
    df[TARGET_COL] = df[TARGET_COL].map(
        lambda x: 1 if str(x).startswith("ckd") else 0
    )

    print(f"[INFO] Loaded dataset: {df.shape}")
    print(f"       CKD={df[TARGET_COL].sum()}  "
          f"non-CKD={(df[TARGET_COL]==0).sum()}")
    return df


# ─────────────────────────────────────────────────────────────────
# 2a. ENCODING
# ─────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every categorical column (except target)."""
    df = df.copy()
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        # Temporarily fill NaN so LabelEncoder doesn't choke
        nan_mask = df[col].isna()
        df[col]  = df[col].fillna("__nan__")
        df[col]  = le.fit_transform(df[col].astype(str))
        df.loc[nan_mask, col] = np.nan   # restore NaN
    print("[INFO] Categorical columns label-encoded.")
    return df


# ─────────────────────────────────────────────────────────────────
# 2b. MISSING VALUE IMPUTATION
# ─────────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Categorical columns → mode imputation
    • Numerical columns   → force numeric, then KNN imputation (k = 5)
    """
    df = df.copy()

    # Force numerical columns to proper float (replaces any leftover '?' with NaN)
    num_present = [c for c in NUMERICAL_COLS if c in df.columns]
    for col in num_present:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Mode imputation for categorical
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        mode_val = df[col].mode(dropna=True)
        if len(mode_val):
            df[col] = df[col].fillna(mode_val[0])

    # KNN imputation for numerical
    imputer = KNNImputer(n_neighbors=5)
    df[num_present] = imputer.fit_transform(df[num_present])

    remaining_nan = df.isnull().sum().sum()
    print(f"[INFO] Missing values handled. Remaining NaN: {remaining_nan}")
    return df


# ─────────────────────────────────────────────────────────────────
# 2c. CUCKOO SEARCH – OUTLIER ADJUSTMENT  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────
def _levy_flight(beta: float, rng: np.random.Generator) -> float:
    """Draw a Lévy-flight step magnitude."""
    num   = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    denom = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / denom) ** (1 / beta)
    u = rng.normal(0, sigma)
    v = rng.normal(0, 1)
    return u / (abs(v) ** (1 / beta))


def cuckoo_search_outlier_adjustment(
        df: pd.DataFrame,
        n_nests: int   = 25,
        max_iter: int  = 100,
        pa: float      = 0.25,
        seed: int      = 42) -> pd.DataFrame:
    """
    Algorithm 1 – Cuckoo Search for Outlier Adjustment.

    For each numerical feature, values beyond ±3 SD are flagged as
    outliers. A population of 'nests' (candidate replacement values) is
    evolved via Lévy flights; the nest with minimum deviation from the
    column mean is applied.
    """
    rng = np.random.default_rng(seed)
    df  = df.copy()
    num_present = [c for c in NUMERICAL_COLS if c in df.columns]

    print("[INFO] Running Cuckoo Search outlier adjustment …")
    total_adjusted = 0

    for col in num_present:
        values = df[col].values.astype(float)
        mu, sigma = np.mean(values), np.std(values)
        if sigma == 0:
            continue

        outlier_mask = np.abs(values - mu) > 3 * sigma
        n_outliers   = outlier_mask.sum()
        if n_outliers == 0:
            continue

        inlier_vals  = values[~outlier_mask]
        outlier_idx  = np.where(outlier_mask)[0]

        # Initialise nests: each nest = array of replacement values
        nests = [rng.choice(inlier_vals, size=n_outliers)
                 for _ in range(n_nests)]

        def fitness(nest):
            # Minimise deviation from column mean (negate for max)
            return -np.mean((nest - mu) ** 2)

        best_nest  = nests[0].copy()
        best_score = fitness(best_nest)

        for _ in range(max_iter):
            for i in range(n_nests):
                step     = _levy_flight(1.5, rng)
                new_nest = nests[i] + 0.01 * step * (nests[i] - mu)
                new_nest = np.clip(new_nest,
                                   mu - 3 * sigma,
                                   mu + 3 * sigma)
                sc = fitness(new_nest)
                if sc > best_score:
                    best_nest  = new_nest.copy()
                    best_score = sc
                    nests[i]   = new_nest

            # Abandon worst fraction
            n_abandon = max(1, int(pa * n_nests))
            for _ in range(n_abandon):
                idx        = rng.integers(0, n_nests)
                nests[idx] = rng.choice(inlier_vals, size=n_outliers)

        values[outlier_idx]  = best_nest
        df[col]              = values
        total_adjusted      += n_outliers

    print(f"[INFO] Cuckoo Search complete. "
          f"Outliers adjusted: {total_adjusted}")
    return df


# ─────────────────────────────────────────────────────────────────
# 2d. ONE-WAY ANOVA – FEATURE SIGNIFICANCE
# ─────────────────────────────────────────────────────────────────
def feature_significance_anova(df: pd.DataFrame):
    """Remove features with ANOVA p-value > 0.05 (paper removes 'pot')."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    f_vals, p_vals = f_classif(X, y)
    anova_df = pd.DataFrame({"feature": X.columns,
                              "f_stat":  f_vals,
                              "p_value": p_vals}).sort_values("p_value")
    non_sig = anova_df[anova_df["p_value"] > 0.05]["feature"].tolist()
    if non_sig:
        print(f"[INFO] ANOVA: removing non-significant features: {non_sig}")
        df = df.drop(columns=non_sig)
    else:
        print("[INFO] ANOVA: all features are significant.")
    return df, anova_df


# ─────────────────────────────────────────────────────────────────
# 2e. STANDARDISATION
# ─────────────────────────────────────────────────────────────────
def standardize(X_train, X_test):
    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ─────────────────────────────────────────────────────────────────
# 2f. SIMULATED ANNEALING – FEATURE SELECTION  (Algorithm 2)
# ─────────────────────────────────────────────────────────────────
def _sa_fitness(X, y, mask):
    """5-fold CV accuracy of Logistic Regression as a fast fitness proxy."""
    if mask.sum() == 0:
        return 0.0
    Xs  = X[:, mask]
    lr  = LogisticRegression(max_iter=500, random_state=42)
    return cross_val_score(lr, Xs, y, cv=5,
                           scoring="accuracy", n_jobs=-1).mean()


def simulated_annealing_feature_selection(
        X, y, feature_names,
        max_iter: int   = 1000,
        T_init:   float = 1.0,
        T_min:    float = 1e-4,
        cooling:  float = 0.995,
        seed:     int   = 42):
    """
    Algorithm 2 – Simulated Annealing for Feature Selection.

    Returns:
        best_mask   – boolean array, True = feature selected
        avg_history – average fitness per iteration
        max_history – best fitness per iteration
    """
    rng        = np.random.default_rng(seed)
    n_features = X.shape[1]

    # Initialise with a random subset
    current = rng.integers(0, 2, size=n_features).astype(bool)
    if not current.any():
        current[rng.integers(0, n_features)] = True

    best        = current.copy()
    best_score  = _sa_fitness(X, y, best)
    curr_score  = best_score
    T           = T_init
    avg_history, max_history = [], []

    print(f"[INFO] Simulated Annealing feature selection "
          f"(max_iter={max_iter}) …")

    for it in range(max_iter):
        # Flip one random bit → neighbour solution
        neighbour          = current.copy()
        flip               = rng.integers(0, n_features)
        neighbour[flip]    = ~neighbour[flip]
        if not neighbour.any():
            neighbour[flip] = True

        nb_score = _sa_fitness(X, y, neighbour)
        delta    = nb_score - curr_score

        # Accept better OR accept worse with probability exp(delta/T)
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current    = neighbour.copy()
            curr_score = nb_score

        if curr_score > best_score:
            best       = current.copy()
            best_score = curr_score

        T = max(T * cooling, T_min)
        avg_history.append(curr_score)
        max_history.append(best_score)

        if (it + 1) % 200 == 0:
            print(f"   iter {it+1:4d} | T={T:.6f} | "
                  f"best={best_score:.4f} | "
                  f"n_selected={int(best.sum())}")

    selected = [feature_names[i] for i, m in enumerate(best) if m]
    print(f"[INFO] SA selected {len(selected)} features: {selected}")
    return best, avg_history, max_history


# ─────────────────────────────────────────────────────────────────
# 3.  SMOTE – DATA BALANCING
# ─────────────────────────────────────────────────────────────────
def balance_smote(X_train, y_train,
                  target_per_class: int = 450,
                  seed: int = 42):
    counts   = pd.Series(y_train).value_counts().to_dict()
    strategy = {k: target_per_class for k in counts}
    sm       = SMOTE(sampling_strategy=strategy, random_state=seed)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────
# 4.  MODEL DEFINITIONS + GRID SEARCH CONFIGS
# ─────────────────────────────────────────────────────────────────
def build_model_configs() -> dict:
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C":      [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "MLP": {
            "model": MLPClassifier(max_iter=500, random_state=42),
            "params": {
                "hidden_layer_sizes": [(100,), (100, 50), (64, 32)],
                "activation":         ["relu", "tanh"],
                "alpha":              [0.0001, 0.001]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators":    [100, 200],
                "max_depth":       [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        # Paper's Table IX parameters for CatBoost
        "CatBoost": {
            "model": CatBoostClassifier(
                random_seed=42, verbose=0,
                border_count=32, depth=8,
                iterations=200, l2_leaf_reg=3,
                learning_rate=0.01
            ),
            "params": {
                "iterations":    [200, 300],
                "depth":         [6, 8],
                "learning_rate": [0.01, 0.05],
                "l2_leaf_reg":   [1, 3]
            }
        }
    }


def train_with_grid_search(configs: dict,
                            X_train, y_train,
                            cv_folds: int = 5) -> dict:
    trained = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, cfg in configs.items():
        print(f"\n[TRAIN] {name} …")
        t0 = time.time()

        gs = GridSearchCV(cfg["model"], cfg["params"],
                          cv=cv, scoring="accuracy",
                          n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        elapsed = time.time() - t0

        cv_scores = cross_val_score(gs.best_estimator_,
                                    X_train, y_train,
                                    cv=cv, scoring="accuracy")

        trained[name] = {
            "best_model":    gs.best_estimator_,
            "best_params":   gs.best_params_,
            "train_accuracy": gs.best_score_,
            "cv_scores":     cv_scores,
            "training_time": elapsed
        }

        print(f"   Best params : {gs.best_params_}")
        print(f"   CV Accuracy : {gs.best_score_:.4f}  "
              f"(±{cv_scores.std():.4f})")
        print(f"   Time        : {elapsed:.2f}s")

    return trained


# ─────────────────────────────────────────────────────────────────
# 5.  EVALUATION
# ─────────────────────────────────────────────────────────────────
def evaluate_all(trained: dict, X_test, y_test) -> dict:
    results = {}

    for name, info in trained.items():
        model  = info["best_model"]
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, "predict_proba") else y_pred.astype(float))

        results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "auc":       roc_auc_score(y_test, y_prob),
            "kappa":     cohen_kappa_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
            "cm":        confusion_matrix(y_test, y_pred),
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "report":    classification_report(y_test, y_pred, output_dict=True)
        }

        print(f"\n[EVAL] {name}")
        print(f"   Test Accuracy : {results[name]['accuracy']*100:.2f}%")
        print(f"   AUC           : {results[name]['auc']:.4f}")
        print(f"   Cohen's Kappa : {results[name]['kappa']:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["non-CKD", "CKD"]))

    return results


# ─────────────────────────────────────────────────────────────────
# 6.  PLOTS
# ─────────────────────────────────────────────────────────────────

def plot_missing_values(df_raw: pd.DataFrame):
    """Figure 1 – Missing Values Count for Each Feature."""
    missing = (df_raw.drop(columns=[TARGET_COL], errors="ignore")
               .isnull().sum().sort_values())
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["#e57373" if v > 50 else "#90caf9" for v in missing.values]
    missing.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Count of Missing Values", fontsize=11)
    ax.set_title("Fig 1 – Missing Values Count for Each Feature", fontsize=12)
    ax.axvline(50, color="red", linestyle="--", lw=1, alpha=0.5,
               label="> 50 missing")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_missing_values.png", dpi=150)
    plt.close()
    print("[PLOT] fig1_missing_values.png saved.")


def plot_sa_fitness(avg_history, max_history):
    """Figure 2 – Fitness Curves During Simulated Annealing."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(avg_history, label="Average Fitness",
            color="darkorange", lw=1, alpha=0.8)
    ax.plot(max_history, label="Maximum Fitness",
            color="steelblue", lw=1.5, linestyle="--")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Fitness (CV Accuracy)", fontsize=11)
    ax.set_title("Fig 2 – Fitness Curves During Simulated Annealing",
                 fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_sa_fitness.png", dpi=150)
    plt.close()
    print("[PLOT] fig2_sa_fitness.png saved.")


def plot_learning_curves(trained: dict, X_train, y_train):
    """Figure 4 – Learning Curves for All Models."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes      = axes.flatten()
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ax, (name, info) in zip(axes, trained.items()):
        sizes, tr_scores, val_scores = learning_curve(
            info["best_model"], X_train, y_train,
            cv=cv, scoring="accuracy",
            train_sizes=np.linspace(0.2, 1.0, 6),
            n_jobs=-1
        )
        ax.plot(sizes, tr_scores.mean(axis=1),  "o-",
                label="Training score",       color="steelblue")
        ax.plot(sizes, val_scores.mean(axis=1), "s--",
                label="Cross-validation score", color="darkorange")
        ax.fill_between(sizes,
                        tr_scores.mean(1)  - tr_scores.std(1),
                        tr_scores.mean(1)  + tr_scores.std(1),
                        alpha=0.15, color="steelblue")
        ax.fill_between(sizes,
                        val_scores.mean(1) - val_scores.std(1),
                        val_scores.mean(1) + val_scores.std(1),
                        alpha=0.15, color="darkorange")
        ax.set_title(f"Learning Curve: {name}", fontsize=10)
        ax.set_xlabel("Training Examples", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylim(0.80, 1.02)

    plt.suptitle("Fig 4 – Learning Curves for All Models", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_learning_curves.png", dpi=150)
    plt.close()
    print("[PLOT] fig4_learning_curves.png saved.")


def plot_confusion_matrices(results: dict):
    """Figure 5 – Confusion Matrix for All Models."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes      = axes.flatten()

    for ax, (name, res) in zip(axes, results.items()):
        cm = res["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    ax=ax, cbar=False,
                    xticklabels=["non-CKD", "CKD"],
                    yticklabels=["non-CKD", "CKD"],
                    annot_kws={"size": 14})
        ax.set_title(f"{name}\nAcc={res['accuracy']*100:.2f}%  "
                     f"AUC={res['auc']:.4f}", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

    plt.suptitle("Fig 5 – Confusion Matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_confusion_matrices.png", dpi=150)
    plt.close()
    print("[PLOT] fig5_confusion_matrices.png saved.")


def plot_roc_curves(results: dict, y_test):
    """Figure 6 – ROC Curves for All Models."""
    colors = {"Logistic Regression": "blue",
              "MLP":                 "green",
              "Random Forest":       "red",
              "CatBoost":            "purple"}
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr,
                label=f"{name} (AUC={res['auc']:.4f})",
                color=colors.get(name, "black"), lw=1.8)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("Fig 6 – ROC Curve for All Models", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_roc_curves.png", dpi=150)
    plt.close()
    print("[PLOT] fig6_roc_curves.png saved.")


def plot_shap(model, X_test_df: pd.DataFrame):
    """Figure 7 – SHAP Summary Plot for CatBoost."""
    print("[INFO] Computing SHAP values …")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)

        # CatBoost may return a list (one array per class) or 2-D array
        if isinstance(shap_values, list):
            sv = shap_values[1]          # class-1 (CKD) SHAP values
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, X_test_df,
                          feature_names=X_test_df.columns.tolist(),
                          show=False, plot_size=None)
        plt.title("Fig 7 – SHAP Summary Plot (CatBoost)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/fig7_shap_summary.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("[PLOT] fig7_shap_summary.png saved.")
    except Exception as exc:
        print(f"[WARN] SHAP plot failed: {exc}")


def plot_anova_pvalues(anova_df: pd.DataFrame):
    """Extra – ANOVA p-values bar chart."""
    df_plot = anova_df.sort_values("p_value")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#e57373" if p > 0.05 else "#81c784"
                for p in df_plot["p_value"]]
    ax.barh(df_plot["feature"], df_plot["p_value"],
            color=colors, edgecolor="white")
    ax.axvline(0.05, color="red", lw=1.5, linestyle="--",
               label="p = 0.05 threshold")
    ax.set_xlabel("p-value", fontsize=11)
    ax.set_title("ANOVA Feature Significance (p-values)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/anova_pvalues.png", dpi=150)
    plt.close()
    print("[PLOT] anova_pvalues.png saved.")


# ─────────────────────────────────────────────────────────────────
# 7.  SUMMARY TABLES  (console)
# ─────────────────────────────────────────────────────────────────
def print_tables(trained: dict, results: dict):
    names = list(trained.keys())
    SEP   = "=" * 72
    header = f"{'Metric':<24}" + "".join(f"{n:>12}" for n in names)

    print(f"\n{SEP}")
    print("TABLE III – Model Training Accuracy & 5-Fold CV Accuracy")
    print(SEP)
    print(header); print("-" * 72)
    print(f"{'Train Accuracy':<24}" +
          "".join(f"{trained[n]['train_accuracy']:>12.4f}" for n in names))
    print(f"{'5-Fold CV Accuracy':<24}" +
          "".join(f"{trained[n]['cv_scores'].mean():>12.4f}" for n in names))

    print(f"\n{SEP}")
    print("TABLE IV – Per-Class Metrics (Test Set)")
    print(SEP)
    for name, res in results.items():
        print(f"\n  {name}")
        rpt = res["report"]
        for cls, label in [("0", "non-CKD"), ("1", "CKD")]:
            if cls in rpt:
                r = rpt[cls]
                print(f"    {label:>8}  "
                      f"Precision={r['precision']:.4f}  "
                      f"Recall={r['recall']:.4f}  "
                      f"F1={r['f1-score']:.4f}")

    print(f"\n{SEP}")
    print("TABLE V – Training Time (seconds)")
    print(SEP)
    print(header); print("-" * 72)
    print(f"{'Training Time (s)':<24}" +
          "".join(f"{trained[n]['training_time']:>12.4f}" for n in names))

    print(f"\n{SEP}")
    print("TABLE VI – Test Accuracy and AUC")
    print(SEP)
    print(header); print("-" * 72)
    print(f"{'Test Accuracy':<24}" +
          "".join(f"{results[n]['accuracy']*100:>11.2f}%" for n in names))
    print(f"{'AUC Score':<24}" +
          "".join(f"{results[n]['auc']:>12.4f}" for n in names))

    print(f"\n{SEP}")
    print("TABLE VII – Per-Class Test Metrics (detailed)")
    print(SEP)
    for name, res in results.items():
        print(f"\n  {name}")
        rpt = res["report"]
        for cls, label in [("0", "non-CKD"), ("1", "CKD")]:
            if cls in rpt:
                r = rpt[cls]
                print(f"    {label:>8}  "
                      f"Precision={r['precision']:.4f}  "
                      f"Recall={r['recall']:.4f}  "
                      f"F1={r['f1-score']:.4f}")

    print(f"\n{SEP}")
    print("TABLE VIII – Cohen's Kappa Scores")
    print(SEP)
    print(header); print("-" * 72)
    kappa_label = "Cohen's Kappa"
    print(f"{kappa_label:<24}" +
          "".join(f"{results[n]['kappa']:>12.4f}" for n in names))

    print(f"\n{SEP}")
    print("TABLE IX – CatBoost Best Parameters")
    print(SEP)
    for k, v in trained["CatBoost"]["best_params"].items():
        print(f"  {k:<20} {v}")
    print(SEP)


# ─────────────────────────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 65)
    print("  CKD Detection – Full Research Paper Pipeline")
    print("  Dataset: kidney_disease.csv")
    print("=" * 65 + "\n")

    # ── Step 1: Load & clean ──────────────────────────────────────
    df_raw = load_and_clean(DATA_PATH)

    # ── Step 2: Plot missing values (Fig 1) ──────────────────────
    plot_missing_values(df_raw)

    # ── Step 3: Encode categoricals ───────────────────────────────
    df = encode_categoricals(df_raw)

    # ── Step 4: Handle missing values ────────────────────────────
    df = handle_missing_values(df)

    # ── Step 5: Cuckoo Search – outlier adjustment ───────────────
    df = cuckoo_search_outlier_adjustment(df)

    # ── Step 6: ANOVA – feature significance ─────────────────────
    df, anova_df = feature_significance_anova(df)
    plot_anova_pvalues(anova_df)

    # ── Step 7: Train / test split (80:20) ───────────────────────
    X            = df.drop(columns=[TARGET_COL]).values
    y            = df[TARGET_COL].values
    feature_names = df.drop(columns=[TARGET_COL]).columns.tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train size: {X_train_raw.shape[0]}  "
          f"Test size: {X_test_raw.shape[0]}")

    # ── Step 8: Standardise ───────────────────────────────────────
    X_train_s, X_test_s, scaler = standardize(X_train_raw, X_test_raw)

    # ── Step 9: Simulated Annealing – feature selection ──────────
    sa_mask, avg_hist, max_hist = simulated_annealing_feature_selection(
        X_train_s, y_train, feature_names,
        max_iter=1000
    )
    plot_sa_fitness(avg_hist, max_hist)  # Fig 2

    X_train_sel   = X_train_s[:, sa_mask]
    X_test_sel    = X_test_s[:, sa_mask]
    sel_features  = [feature_names[i]
                     for i, m in enumerate(sa_mask) if m]
    print(f"[INFO] Selected features ({len(sel_features)}): {sel_features}")

    # ── Step 10: SMOTE – balance training set ────────────────────
    X_train_bal, y_train_bal = balance_smote(X_train_sel, y_train)

    # ── Step 11: Train all models ─────────────────────────────────
    configs = build_model_configs()
    trained = train_with_grid_search(configs, X_train_bal, y_train_bal)

    # ── Step 12: Learning curves (Fig 4) ─────────────────────────
    plot_learning_curves(trained, X_train_bal, y_train_bal)

    # ── Step 13: Evaluate on test set ────────────────────────────
    results = evaluate_all(trained, X_test_sel, y_test)

    # ── Step 14: Plots (Figs 5, 6) ───────────────────────────────
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)

    # ── Step 15: SHAP for CatBoost (Fig 7) ───────────────────────
    catboost_model = trained["CatBoost"]["best_model"]
    X_test_df      = pd.DataFrame(X_test_sel, columns=sel_features)
    plot_shap(catboost_model, X_test_df)

    # ── Step 16: Print summary tables ────────────────────────────
    print_tables(trained, results)

    print(f"\n[DONE] All outputs saved to {OUTPUT_DIR}/")
    return trained, results


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()