import pandas as pd
from timeseries_toolkit import DataDriftDetector


def test_drift():
    dfA = pd.read_csv("./data/sample_dataset.csv")
    dfB = pd.read_csv("./data/sample_dataset_drifted.csv")
    detector = DataDriftDetector(drift_threshold=0.7, random_state=42)
    results = detector.detect_drift(
        reference_data=dfA,
        comparison_data=dfB,
        categorical_columns=["city","weather_condition","event"],
        verbose=True
    )
    summary = detector.get_drift_summary()
    print("\nDrift Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    detector.export_results('drift_results.json', format='json')