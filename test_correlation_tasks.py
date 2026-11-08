import numpy as np
import pandas as pd
import pytest
from grader_contracts.correlation_tasks import BrainDataInput
from correlation_tasks import analyze_brain_correlations, FEATURES


def make_file(tmp_path, with_na=True):
    rng = np.random.default_rng(0)
    rows = []
    for gender, n in (("Female", 12), ("Male", 10)):
        mri = rng.normal(900000, 50000, n)
        for i in range(n):
            rows.append({
                "Gender": gender,
                "FSIQ": rng.normal(110, 10),
                "VIQ": rng.normal(110, 10),
                "PIQ": rng.normal(110, 10),
                "Weight": 100 + mri[i] / 10000,
                "Height": rng.normal(68, 3),
                "MRI_Count": mri[i],
            })
    df = pd.DataFrame(rows)
    if with_na:
        df.loc[3, "Weight"] = np.nan
        df.loc[15, "Height"] = np.nan
    path = tmp_path / "brainsize.txt"
    df.to_csv(path, sep="\t", index=False, na_rep="NA")
    return str(path), df


def test_counts_and_keys(tmp_path):
    path, _ = make_file(tmp_path)
    r = analyze_brain_correlations(BrainDataInput(path))
    assert (r.women_count, r.men_count) == (12, 10)
    assert set(r.women_mri_correlation) == set(FEATURES) == set(r.men_mri_correlation)
    assert "MRI_Count" not in r.women_mri_correlation


def test_values_match_numpy_with_na(tmp_path):
    path, df = make_file(tmp_path)
    r = analyze_brain_correlations(BrainDataInput(path))
    women = df[df.Gender == "Female"].dropna(subset=["Weight", "MRI_Count"])
    expected = np.corrcoef(women["Weight"], women["MRI_Count"])[0, 1]
    assert r.women_mri_correlation["Weight"] == pytest.approx(expected)
    men = df[df.Gender == "Male"].dropna(subset=["Height", "MRI_Count"])
    expected = np.corrcoef(men["Height"], men["MRI_Count"])[0, 1]
    assert r.men_mri_correlation["Height"] == pytest.approx(expected)


def test_strongest_feature(tmp_path):
    path, _ = make_file(tmp_path)
    r = analyze_brain_correlations(BrainDataInput(path))
    assert r.strongest_mri_feature == "Weight"
