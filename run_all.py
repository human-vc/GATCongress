import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import SEED
import numpy as np


def main():
    np.random.seed(SEED)

    print("=" * 60)
    print("CongressNetwork Pipeline")
    print("=" * 60)

    print("\n[1/10] Data Pipeline")
    from data_pipeline import main as run_data
    run_data()

    print("\n[2/10] Spectral Analysis")
    from spectral_analysis import main as run_spectral
    run_spectral()

    print("\n[3/10] BLI Regression")
    from bli_regression import main as run_bli
    run_bli()

    print("\n[4/10] Freshman Cohort Analysis")
    from freshman_cohort_analysis import main as run_freshman
    run_freshman()

    print("\n[5/10] Null Model Analysis")
    from null_model_analysis import main as run_null
    run_null()

    print("\n[6/10] Weighted Spectral Analysis")
    from weighted_spectral import main as run_weighted
    run_weighted()

    print("\n[7/10] Vote Filtering Analysis")
    from vote_filtering import main as run_filtering
    run_filtering()

    print("\n[8/10] Recovery Threshold Sensitivity")
    from recovery_threshold_sensitivity import main as run_recovery
    run_recovery()

    print("\n[9/10] Counterfactual Sensitivity")
    from counterfactual_sensitivity import main as run_counterfactual
    run_counterfactual()

    print("\n[10/10] Figures")
    from generate_figures import main as run_figures
    run_figures()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
