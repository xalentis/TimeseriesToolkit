import pandas as pd
from timeseries_toolkit import TimeSeriesFeatureEngineering
from timeseries_toolkit import AutoLSTM


def test_autolstm():
    df = pd.read_csv("./data/bitcoin.csv")
    
    # engineer some features
    fe = TimeSeriesFeatureEngineering(correlation_threshold=0.9, importance_threshold=0.01, categorical_threshold=10)
    engineered_df, _ = fe.engineer_features(
        df=df,
        target_column='close',
        date_columns=['timeOpen'],
        auto_detect_lags=True,
        make_stationary=False,
        remove_seasonal=False,
        create_seasonal_indicators=True,
        feature_selection=True,
        scale_features=True,
        max_features=100,
        verbose=True
    )

    # autolstm don't like date objects
    engineered_df.drop(columns=["timeOpen"], inplace=True)
    
    # initialize and train AutoLSTM
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
    new_data = engineered_df.drop(columns=['close']).tail(20)  # Last 20 rows
    predictions = auto_lstm.predict(new_data)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5].flatten()}")