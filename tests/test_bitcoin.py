import pandas as pd
from timeseries_toolkit import TimeSeriesFeatureEngineering
from timeseries_toolkit import TimeSeriesAnomalyDetector


def test_bitcoin_fe():
    fe = TimeSeriesFeatureEngineering(correlation_threshold=0.9, importance_threshold=0.01, categorical_threshold=10)

    df = pd.read_csv("./data/bitcoin.csv")
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


def test_bitcoin_anomaly():
    df = pd.read_csv("./data/bitcoin.csv")
    detector = TimeSeriesAnomalyDetector()
    detector.load_data(df, date_col='timeOpen')
    detector.preprocess(fill_method='interpolate')
    detector.detect_anomalies(methods=['zscore', 'iqr', 'isolation_forest', 'seasonal'])
    summary = detector.get_anomaly_summary()
    print(summary)
    anomalies = detector.get_anomalies()
    print(anomalies[['date', 'column', 'value', 'anomaly_severity', 'severity_category']])