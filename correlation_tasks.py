"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import math
import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
NA_VALUES = ["NA", "N/A", "nan", "NaN", ".", ""]


def _load_brain_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", na_values=NA_VALUES)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python", na_values=NA_VALUES)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    df = df.rename(columns={"MRICount": "MRI_Count", "MRI Count": "MRI_Count"})
    return df


def _gender_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in ("gender", "sex"):
            return col
    raise KeyError("В таблице нет столбца с полом (Gender)")


def _correlations(group: pd.DataFrame) -> dict[str, float]:
    return {f: float(group[f].corr(group["MRI_Count"], method="pearson")) for f in FEATURES}


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = _load_brain_data(data.csv_path)
    for col in FEATURES + ["MRI_Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    gender = df[_gender_column(df)].astype(str).str.strip().str.lower()
    women = df[gender.str.startswith("f")]
    men = df[gender.str.startswith("m")]

    women_corr = _correlations(women)
    men_corr = _correlations(men)

    best_feature, best_abs = "", -1.0
    for corr in (women_corr, men_corr):
        for feature, value in corr.items():
            if not math.isnan(value) and abs(value) > best_abs:
                best_feature, best_abs = feature, abs(value)

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=best_feature,
    )


def plot_brain_correlations(csv_path: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = _load_brain_data(csv_path)
    gender_col = _gender_column(df)

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 4))
    for ax, feature in zip(axes, FEATURES):
        sns.scatterplot(data=df, x=feature, y="MRI_Count", hue=gender_col, ax=ax)
    plt.tight_layout()
    plt.show()

    for name, group in df.groupby(gender_col):
        plt.figure(figsize=(6, 5))
        sns.heatmap(group[FEATURES + ["MRI_Count"]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Корреляции: {name}")
        plt.show()
