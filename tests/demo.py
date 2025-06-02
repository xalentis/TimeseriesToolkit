# Building the package using Poetry:
# poetry build

# Installation after build:
# pip install timeseries_toolkit-0.1.0-py3-none-any.whl

# Run unit tests:
# poetry run pytest -v -s --tb=long


# Example usage:
import pandas as pd
from timeseries_toolkit import TimeSeriesAnomalyDetector
from timeseries_toolkit import TimeSeriesFeatureEngineering
from timeseries_toolkit import DataDriftDetector
from timeseries_toolkit import TimeSeriesSuitabilityScorer
from timeseries_toolkit import AutoLSTM

# load a dataset
df = pd.read_csv("./data/bitcoin.csv")

# using the anomaly detector
detector = TimeSeriesAnomalyDetector()
detector.load_data(df, date_col='timeOpen')
detector.preprocess(fill_method='interpolate')
detector.detect_anomalies(methods=['zscore', 'iqr', 'isolation_forest', 'seasonal'])
detector.plot_seasonal_decomposition('close')
detector.plot_distribution('close')
summary = detector.get_anomaly_summary()
print(summary)
anomalies = detector.get_anomalies()
print(anomalies[['date', 'column', 'value', 'anomaly_severity', 'severity_category']])

# determine how suitable it is for time-series forecasting
scorer = TimeSeriesSuitabilityScorer(min_observations=10)
result = scorer.score(df, date_column='timeOpen', value_columns=["open","high","low","close","volume","marketCap"])
print(f"Score: {result['overall_score']:.1f}/100")
print(f"Suitable: {result['is_timeseries_suitable']}")
print("Issues:", result['issues'])

# auto-engineer features
fe = TimeSeriesFeatureEngineering(correlation_threshold=0.9, importance_threshold=0.01, categorical_threshold=10)
engineered_df, metadata = fe.engineer_features(
    df=df,
    target_column='close',
    date_columns=['timeOpen'],
    auto_detect_lags=True,
    make_stationary=False,
    remove_seasonal=False,
    create_seasonal_indicators=True,
    feature_selection=True,
    scale_features=True,
    max_features=20,
    verbose=True
)

# remove date objects for auto-lstm
engineered_df.drop(columns=["timeOpen"], inplace=True)

# auto-generate an LSTM architecture
auto_lstm = AutoLSTM(
    task_type='sequence_prediction',
    max_epochs=100,
    patience=5,
    verbose=True,
    use_dropout=False
)

# fit the model
auto_lstm.fit(engineered_df, target_column='close')

# train the model
history = auto_lstm.train()
print(f"Training completed in {history['epochs_trained']} epochs")
print(f"Best validation loss: {history['best_val_loss']:.4f}")

# evaluate the model
metrics = auto_lstm.evaluate()
print(f"Test Metrics: {metrics}")

# make predictions on new data
new_data = engineered_df.drop(columns=['close']).tail(20)
predictions = auto_lstm.predict(new_data)
print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:5].flatten()}")

# Using the data drift detector
dfA = pd.read_csv("./data/sample_dataset.csv")
dfB = pd.read_csv("./data/sample_dataset_drifted.csv")

detector = DataDriftDetector(drift_threshold=0.7, random_state=42)
results = detector.detect_drift(reference_data=dfA, comparison_data=dfB, categorical_columns=["city","weather_condition","event"], verbose=True)
summary = detector.get_drift_summary()
print("\nDrift Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")
