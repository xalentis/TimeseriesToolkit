
import pandas as pd
from timeseries_toolkit import TimeSeriesAnomalyDetector
from timeseries_toolkit import TimeSeriesFeatureEngineering
from timeseries_toolkit import DataDriftDetector
from timeseries_toolkit import TimeSeriesSuitabilityScorer

# Build:
# poetry build

# Install:
# pip install timeseries_toolkit-0.1.0-py3-none-any.whl

# Unit Tests:
# poetry run pytest -v -s --tb=long

# Using the time series suitability scorer:
df = pd.read_csv("./data/bitcoin.csv")
scorer = TimeSeriesSuitabilityScorer(min_observations=10)
result = scorer.score(df, date_column='timeOpen', value_columns=["open","high","low","close","volume","marketCap"])
print(f"Score: {result['overall_score']:.1f}/100")
print(f"Suitable: {result['is_timeseries_suitable']}")
print("Issues:", result['issues'])

# Using auto-feature engineering
fe = TimeSeriesFeatureEngineering(correlation_threshold=0.9, importance_threshold=0.01, categorical_threshold=10)
df = pd.read_csv("./data/AirQualityUCI.csv", sep=";")
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
df.dropna(axis=0, inplace=True)
engineered_df, metadata = fe.engineer_features(
    df=df,
    target_column='CO(GT)',
    date_columns=['Date'],
    auto_detect_lags=True,
    make_stationary=False,
    remove_seasonal=False,
    create_seasonal_indicators=True,
    feature_selection=True,
    scale_features=True,
    max_features=30,
    verbose=True
)
print(f"Original features: {len(df.columns)}")
print(f"Final features: {len(engineered_df.columns)}")
print(f"Created features: {len(metadata['created_features'])}")
summary = fe.get_feature_summary()
print(f"Top 5 Important Features:")
if summary['feature_importance_scores']:
    top_features = sorted(summary['feature_importance_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, score in top_features:
        print(f"  {feat}: {score:.4f}")

# Using the anomaly detector
detector = TimeSeriesAnomalyDetector()
detector.load_data(df, date_col='Date')
detector.preprocess(fill_method='interpolate')
detector.detect_anomalies(methods=['zscore', 'iqr', 'isolation_forest', 'seasonal'])
detector.plot_seasonal_decomposition('Demand Forecast')
detector.plot_distribution('Demand Forecast')
summary = detector.get_anomaly_summary()
print(summary)
anomalies = detector.get_anomalies()
print(anomalies[['date', 'column', 'value', 'anomaly_severity', 'severity_category']])


# Using the data drift detector
dfA = pd.read_csv("./data/sample_dataset.csv")
dfB = pd.read_csv("./data/sample_dataset_drifted.csv")

detector = DataDriftDetector(drift_threshold=0.7, random_state=42)
results = detector.detect_drift(reference_data=dfA, comparison_data=dfB, categorical_columns=["city","weather_condition","event"], verbose=True)
summary = detector.get_drift_summary()
print("\nDrift Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
detector.export_results('drift_results.json', format='json')