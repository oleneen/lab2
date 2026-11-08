"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    df = pd.read_csv(data.csv_path)

    missing_by_column = {str(col): int(n) for col, n in df.isna().sum().items()}
    adults_over_30_count = int((df["Age"] > 30).sum())

    mean_age = df.groupby("Pclass")["Age"].mean()
    survival = df.groupby("Pclass")["Survived"].mean()

    return TitanicSummary(
        row_count=int(len(df)),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass={int(k): float(v) for k, v in mean_age.items()},
        survival_rate_by_pclass={int(k): float(v) for k, v in survival.items()},
        highest_fares=[float(x) for x in df["Fare"].nlargest(5)],
    )
