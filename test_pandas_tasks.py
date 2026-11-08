import pytest
from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic

CSV = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"A, Mr. X",male,22,1,0,T1,7.25,,S
2,1,1,"B, Mrs. Y",female,38,1,0,T2,71.5,C85,C
3,1,3,"C, Miss. Z",female,30,0,0,T3,8.0,,S
4,1,1,"D, Mrs. W",female,,1,0,T4,53.0,C1,S
5,0,3,"E, Mr. V",male,40,0,0,T5,9.0,,S
6,0,2,"F, Mr. U",male,31,0,0,T6,13.0,,Q
7,1,2,"G, Mr. T",male,,0,0,T7,14.0,,S
"""


@pytest.fixture
def summary(tmp_path):
    p = tmp_path / "titanic.csv"
    p.write_text(CSV, encoding="utf-8")
    return analyze_titanic(TitanicInput(str(p)))


def test_rows_and_missing(summary):
    assert summary.row_count == 7
    assert summary.missing_by_column["Age"] == 2
    assert summary.missing_by_column["Cabin"] == 5
    assert summary.missing_by_column["Fare"] == 0
    assert all(isinstance(v, int) for v in summary.missing_by_column.values())


def test_adults_over_30_is_strict(summary):
    assert summary.adults_over_30_count == 3


def test_mean_age_skips_nan(summary):
    assert summary.mean_age_by_pclass == {1: 38.0, 2: 31.0, 3: pytest.approx((22 + 30 + 40) / 3)}
    assert all(isinstance(k, int) for k in summary.mean_age_by_pclass)


def test_survival_rate(summary):
    assert summary.survival_rate_by_pclass == {1: 1.0, 2: 0.5, 3: pytest.approx(1 / 3)}
    assert all(isinstance(k, int) for k in summary.survival_rate_by_pclass)


def test_highest_fares(summary):
    assert summary.highest_fares == [71.5, 53.0, 14.0, 13.0, 9.0]
