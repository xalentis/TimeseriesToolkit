import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

class TimeSeriesAnomalyDetector:
    def __init__(self, df=None, date_col=None, value_cols=None, 
                 resample_freq=None, threshold_zscore=3.0, 
                 threshold_iqr=1.5, contamination=0.05):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing time series data
        date_col : str
            The name of the date column
        value_cols : list
            List of numeric column names to analyze
        resample_freq : str
            Frequency for resampling (e.g., 'D' for daily, 'H' for hourly)
        threshold_zscore : float
            Z-score threshold for anomaly detection
        threshold_iqr : float
            IQR multiplier threshold for anomaly detection
        contamination : float
            Expected proportion of anomalies (for Isolation Forest)
        """
        self.df = df.copy() if df is not None else None
        self.date_col = date_col
        self.value_cols = value_cols
        self.resample_freq = resample_freq
        self.threshold_zscore = threshold_zscore
        self.threshold_iqr = threshold_iqr
        self.contamination = contamination
        self.anomalies = {}
        self.results = {}
        

    def load_data(self, df, date_col, value_cols=None):
        """
        Load data into the detector with automatic date format detection.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing time series data
        date_col : str
            The name of the date column
        value_cols : list, optional
            List of numeric column names to analyze
        """

        
        self.df = df.copy()
        self.date_col = date_col
        
        # auto-detect and convert date format
        self.df[self.date_col] = self._parse_dates(self.df[self.date_col])
        
        # if value_cols not specified, use all numeric columns except the date
        if value_cols is None:
            self.value_cols = [col for col in df.select_dtypes(include=['number']).columns]
        else:
            self.value_cols = value_cols
            
        # sort by date if not already in sequence
        self.df = self.df.sort_values(by=self.date_col).reset_index(drop=True)
        return self

    def _parse_dates(self, date_series):
        """
        Automatically detect and parse various date formats.
        
        Parameters:
        -----------
        date_series : pandas.Series
            Series containing dates in various formats
            
        Returns:
        --------
        pandas.Series
            Series with datetime objects
        """
        
        # if already datetime, return as-is
        if pd.api.types.is_datetime64_any_dtype(date_series):
            return date_series
        
        # get a sample of non-null values for format detection
        sample_values = date_series.dropna().head(10).astype(str)
        
        if len(sample_values) == 0:
            raise ValueError("No valid date values found in the date column")
        
        # check for unix timestamps (numeric values)
        try:
            # try to convert to numeric
            numeric_values = pd.to_numeric(date_series, errors='coerce')
            if not numeric_values.isna().all():
                # ceck if values are in reasonable unix timestamp range
                min_val, max_val = numeric_values.min(), numeric_values.max()
                
                # unix timestamp ranges (approximately)
                if 0 <= min_val and max_val <= 2147483647:  # Unix seconds
                    return pd.to_datetime(numeric_values, unit='s')
                elif 0 <= min_val and max_val <= 2147483647000:  # Unix milliseconds
                    return pd.to_datetime(numeric_values, unit='ms')
        except (ValueError, TypeError, OverflowError):
            pass
        
        # define common date formats to try
        date_formats = [
            # ISO formats
            '%Y-%m-%dT%H:%M:%S.%fZ',      # 2024-04-24T00:00:00.000Z
            '%Y-%m-%dT%H:%M:%SZ',         # 2024-04-24T00:00:00Z
            '%Y-%m-%dT%H:%M:%S',          # 2024-04-24T00:00:00
            '%Y-%m-%d %H:%M:%S.%f',       # 2024-04-24 00:00:00.000
            '%Y-%m-%d %H:%M:%S',          # 2024-04-24 00:00:00
            '%Y-%m-%d',                   # 2024-04-24
            
            # US formats
            '%m/%d/%Y %H:%M:%S',          # 04/24/2024 00:00:00
            '%m/%d/%Y',                   # 04/24/2024
            '%m-%d-%Y',                   # 04-24-2024
            
            # European formats
            '%d/%m/%Y %H:%M:%S',          # 24/04/2024 00:00:00
            '%d/%m/%Y',                   # 24/04/2024
            '%d-%m-%Y',                   # 24-04-2024
            
            # Other common formats
            '%Y/%m/%d',                   # 2024/04/24
            '%d.%m.%Y',                   # 24.04.2024
            '%B %d, %Y',                  # April 24, 2024
            '%b %d, %Y',                  # Apr 24, 2024
        ]
        
        # try each
        for fmt in date_formats:
            try:
                test_sample = sample_values.iloc[0]
                datetime.strptime(test_sample, fmt)
                return pd.to_datetime(date_series, format=fmt, errors='coerce')
            except (ValueError, TypeError):
                continue
        
        # try pandas parser
        try:
            parsed_dates = pd.to_datetime(date_series, errors='coerce', infer_datetime_format=True)
            success_rate = (parsed_dates.notna().sum() / len(parsed_dates))
            if success_rate > 0.8:  # at least 80% successful parsing
                return parsed_dates
            else:
                raise ValueError(f"Date parsing success rate too low: {success_rate:.2%}")
                
        except Exception as e:
            # last resort
            try:
                return pd.to_datetime(date_series, errors='coerce', dayfirst=True)
            except:
                pass
        
        # if all else fails
        sample_str = ', '.join(sample_values.head(3).tolist())
        raise ValueError(
            f"Unable to automatically detect date format. "
            f"Sample values: {sample_str}. "
            f"Please ensure the date column contains valid date strings or unix timestamps."
        )
    

    def preprocess(self, fill_method='interpolate', resample_freq=None):
        """
        Preprocess the data if required by handling missing values and optionally resampling.
        
        Parameters:
        -----------
        fill_method : str
            Method to fill missing values ('interpolate', 'ffill', 'bfill', 'mean')
        resample_freq : str
            Frequency for resampling (e.g., 'D' for daily, 'H' for hourly)
        """
        
        if resample_freq is not None:
            self.resample_freq = resample_freq
            
        df_processed = self.df.copy()
        
        # handle nan values in the date column before setting as index
        original_length = len(df_processed)
        nan_dates = df_processed[self.date_col].isna().sum()
        
        if nan_dates > 0:
            warnings.warn(f"Found {nan_dates} NaN values in date column '{self.date_col}'. "
                        f"These rows will be dropped before processing.")
            df_processed = df_processed.dropna(subset=[self.date_col])
            print(f"Dropped {original_length - len(df_processed)} rows with invalid dates")
        
        # now set the date column as index
        df_processed = df_processed.set_index(self.date_col)
        
        # check for duplicate dates in index
        if df_processed.index.duplicated().any():
            duplicate_count = df_processed.index.duplicated().sum()
            warnings.warn(f"Found {duplicate_count} duplicate dates. Keeping first occurrence of each date.")
            df_processed = df_processed[~df_processed.index.duplicated(keep='first')]
        
        # sort by index to ensure proper time order
        df_processed = df_processed.sort_index()
            
        # resample if specified
        if hasattr(self, 'resample_freq') and self.resample_freq is not None:
            print(f"Resampling data to {self.resample_freq} frequency")
            df_processed = df_processed.resample(self.resample_freq).mean()
            
        # handle missing values in value columns
        if fill_method == 'interpolate':
            if not df_processed.index.is_monotonic_increasing:
                warnings.warn("Date index is not monotonic. Sorting before interpolation.")
                df_processed = df_processed.sort_index()

            try:
                df_processed[self.value_cols] = df_processed[self.value_cols].interpolate(
                    method='time', limit_direction='both'
                )
            except (ValueError, TypeError) as e:
                warnings.warn(f"Time-based interpolation failed ({str(e)}). Using linear interpolation instead.")
                df_processed[self.value_cols] = df_processed[self.value_cols].interpolate(
                    method='linear', limit_direction='both'
                )
                
        elif fill_method == 'ffill':
            df_processed[self.value_cols] = df_processed[self.value_cols].fillna(method='ffill')
            
        elif fill_method == 'bfill':
            df_processed[self.value_cols] = df_processed[self.value_cols].fillna(method='bfill')
            
        elif fill_method == 'mean':
            for col in self.value_cols:
                col_mean = df_processed[col].mean()
                if pd.isna(col_mean):
                    warnings.warn(f"Column '{col}' has no valid values. Cannot fill with mean.")
                else:
                    df_processed[col] = df_processed[col].fillna(col_mean)
                    
        elif fill_method == 'drop':
            before_drop = len(df_processed)
            df_processed = df_processed.dropna(subset=self.value_cols)
            dropped = before_drop - len(df_processed)
            if dropped > 0:
                print(f"Dropped {dropped} rows with missing values in value columns")
        
        df_processed = df_processed.reset_index()
        self.date_col = df_processed.columns[0]
        self.df_processed = df_processed

        print(f"Preprocessing complete. Final dataset: {len(df_processed)} rows, "
            f"date range: {df_processed[self.date_col].min()} to {df_processed[self.date_col].max()}")
        
        missing_summary = df_processed[self.value_cols].isnull().sum()
        if missing_summary.sum() > 0:
            print("Remaining missing values by column:")
            for col, missing in missing_summary.items():
                if missing > 0:
                    print(f"  {col}: {missing} missing values")
        
        return self
    

    def detect_anomalies(self, methods=None, residual_threshold=2):
        """
        Detect anomalies using multiple methods.
        
        Parameters:
        -----------
        methods : list
            List of methods to use ('zscore', 'iqr', 'isolation_forest', 'seasonal')
        
        Returns:
        --------
        dict: Dictionary of dataframes with anomaly flags for each method and column
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'isolation_forest', 'seasonal']
        self.anomalies = {}
        
        df = self.df_processed.copy()
        results = {}
        for col in self.value_cols:
            results[col] = pd.DataFrame({
                'date': df[self.date_col],
                'value': df[col]
            })
            results[col]['zscore_anomaly'] = False
            results[col]['iqr_anomaly'] = False
            results[col]['iforest_anomaly'] = False
            results[col]['seasonal_anomaly'] = False
            results[col]['is_anomaly'] = False
            results[col]['zscore_severity'] = 0.0
            results[col]['iqr_severity'] = 0.0
            results[col]['iforest_severity'] = 0.0
            results[col]['seasonal_severity'] = 0.0
            results[col]['anomaly_severity'] = 0.0
        
        # z-score
        if 'zscore' in methods:
            for col in self.value_cols:
                data = df[col].values
                z_scores = stats.zscore(data, nan_policy='omit')
                z_scores_abs = np.abs(z_scores)
                anomalies = (z_scores_abs > self.threshold_zscore)
                
                # calculate severity - 0 is lowest, 10 is highest
                severity = np.zeros_like(z_scores_abs, dtype=float)
                severity_mask = z_scores_abs > self.threshold_zscore
                max_z = max(10 * self.threshold_zscore, z_scores_abs.max())
                severity[severity_mask] = 1 + 9 * (z_scores_abs[severity_mask] - self.threshold_zscore) / (max_z - self.threshold_zscore)
                severity = np.clip(severity, 0, 10)
                results[col]['zscore'] = z_scores
                results[col]['zscore_anomaly'] = anomalies
                results[col]['zscore_severity'] = severity
                results[col]['is_anomaly'] |= anomalies
                
        # iqr
        if 'iqr' in methods:
            for col in self.value_cols:
                data = df[col].values
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (self.threshold_iqr * IQR)
                upper_bound = Q3 + (self.threshold_iqr * IQR)
                distance_lower = np.maximum(0, lower_bound - data)
                distance_upper = np.maximum(0, data - upper_bound)
                distance = np.maximum(distance_lower, distance_upper)
                
                # mark anomalies
                anomalies = ((data < lower_bound) | (data > upper_bound))
                
                # calculate severity - 1-10 
                severity = np.zeros_like(data, dtype=float)
                if anomalies.any():
                    max_distance = max(distance[anomalies])
                    severity[anomalies] = 1 + 9 * (distance[anomalies] / max_distance)
                results[col]['iqr_lower'] = lower_bound
                results[col]['iqr_upper'] = upper_bound
                results[col]['iqr_anomaly'] = anomalies
                results[col]['iqr_severity'] = severity
                results[col]['is_anomaly'] |= anomalies
                
        # isolation forest
        if 'isolation_forest' in methods:
            for col in self.value_cols:
                data = df[[col]].values
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                model = IsolationForest(contamination=self.contamination, random_state=42)
                predictions = model.fit_predict(data_scaled)
                anomalies = (predictions == -1)
                anomaly_scores = model.decision_function(data_scaled)
                severity = np.zeros_like(anomaly_scores, dtype=float)
                if anomalies.any():
                    min_score = anomaly_scores.min()
                    threshold_score = np.percentile(anomaly_scores, 100 * 0.1)
                    mask = anomalies.flatten()
                    severity[mask] = 1 + 9 * (threshold_score - anomaly_scores[mask]) / (threshold_score - min_score)
                results[col]['iforest_score'] = anomaly_scores
                results[col]['iforest_anomaly'] = anomalies
                results[col]['iforest_severity'] = severity
                results[col]['is_anomaly'] |= anomalies
        
        # seasonal decomposition
        if 'seasonal' in methods:
            for col in self.value_cols:
                try:
                    # need at least 2 full seasonal cycles for decomposition
                    min_periods = 4
                    if len(df) >= min_periods:
                        temp_df = df.set_index(self.date_col)[[col]]
                        
                        # determine period
                        if len(temp_df) < 14:
                            period = 2  # minimal
                        elif len(temp_df) < 60:
                            period = 7  # weekly
                        else:
                            period = 30  # monthly
                        decomposition = seasonal_decompose(
                            temp_df, 
                            model='additive', 
                            period=period,
                            extrapolate_trend=True
                        )
                        residuals = decomposition.resid.abs()
                        mean_residual = residuals.mean()
                        std_residual = residuals.std()
                        threshold = mean_residual + residual_threshold * std_residual
                        seasonal_anomalies = (residuals > threshold).values.flatten()
                        severity = np.zeros_like(seasonal_anomalies, dtype=float)
                        if seasonal_anomalies.any():
                            residual_values = residuals.values.flatten()
                            max_residual = max(residual_values[seasonal_anomalies])
                            severity_mask = seasonal_anomalies
                            severity[severity_mask] = 1 + 9 * (residual_values[severity_mask] - threshold) / (max_residual - threshold)
                        results[col]['seasonal_residual'] = residuals.values
                        results[col]['seasonal_threshold'] = threshold
                        results[col]['seasonal_anomaly'] = seasonal_anomalies
                        results[col]['seasonal_severity'] = severity
                        results[col]['is_anomaly'] |= seasonal_anomalies
                    else:
                        results[col]['seasonal_anomaly'] = False
                        results[col]['seasonal_residual'] = np.nan
                        results[col]['seasonal_threshold'] = np.nan
                        results[col]['seasonal_severity'] = 0.0
                except Exception as e:
                    print(f"Seasonal decomposition failed for column {col}: {e}")
                    results[col]['seasonal_anomaly'] = False
                    results[col]['seasonal_residual'] = np.nan
                    results[col]['seasonal_threshold'] = np.nan
                    results[col]['seasonal_severity'] = 0.0
        
        for col in self.value_cols:
            data = df[col].values
            results[col]['skew'] = stats.skew(data, nan_policy='omit')
            results[col]['kurtosis'] = stats.kurtosis(data, nan_policy='omit')
            method_severities = [
                results[col]['zscore_severity'], 
                results[col]['iqr_severity'],
                results[col]['iforest_severity'],
                results[col]['seasonal_severity']
            ]
            
            # combine severities - take maximum across methods
            results[col]['anomaly_severity'] = np.maximum.reduce([
                np.array(severity) for severity in method_severities
            ])
            results[col]['anomaly_severity'] = np.round(results[col]['anomaly_severity'], 1)
        self.results = results
        return results
    

    def get_anomalies(self):
        """
        Return a dataframe with all anomalies and their severity scores.
        
        Returns:
        --------
        pandas.DataFrame: Combined anomalies across all methods and columns with severity scores
        """
        all_anomalies = []
        
        for col, result_df in self.results.items():
            anomalies = result_df[result_df['is_anomaly']].copy()
            anomalies['column'] = col
            severity = anomalies['anomaly_severity']
            conditions = [
                (severity == 0),
                (severity > 0) & (severity < 3),
                (severity >= 3) & (severity < 6),
                (severity >= 6) & (severity < 8),
                (severity >= 8)
            ]
            choices = ['None', 'Low', 'Medium', 'High', 'Critical']
            anomalies['severity_category'] = np.select(conditions, choices, default='Unknown')
            all_anomalies.append(anomalies)
            
        if all_anomalies:
            combined = pd.concat(all_anomalies, ignore_index=True)
            return combined.sort_values(by='anomaly_severity', ascending=False)
        else:
            return pd.DataFrame()
    

    def plot_anomalies(self, columns=None, methods=None, figsize=(15, 8), show_severity=True):
        """
        Plot the time series data with detected anomalies and their severity.
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to plot (defaults to all value_cols)
        methods : list, optional
            List of anomaly detection methods to highlight
        figsize : tuple
            Figure size
        show_severity : bool
            Whether to show severity in the plot
        """
        if columns is None:
            columns = self.value_cols
        if methods is None:
            methods = ['zscore', 'iqr', 'iforest', 'seasonal']
            
        for col in columns:
            if col in self.results:
                result = self.results[col]
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(result['date'], result['value'], 'b-', label='Raw data')
                if 'zscore' in methods and 'zscore_anomaly' in result:
                    zscore_anomalies = result[result['zscore_anomaly']]
                    ax.scatter(zscore_anomalies['date'], zscore_anomalies['value'], 
                              color='red', marker='o', s=50, label='Z-score anomalies')
                if 'iqr' in methods and 'iqr_anomaly' in result:
                    iqr_anomalies = result[result['iqr_anomaly']]
                    ax.scatter(iqr_anomalies['date'], iqr_anomalies['value'], 
                              color='green', marker='s', s=50, label='IQR anomalies')
                if 'iforest' in methods and 'iforest_anomaly' in result:
                    iforest_anomalies = result[result['iforest_anomaly']]
                    ax.scatter(iforest_anomalies['date'], iforest_anomalies['value'], 
                              color='purple', marker='^', s=50, label='Isolation Forest anomalies')
                if 'seasonal' in methods and 'seasonal_anomaly' in result:
                    seasonal_anomalies = result[result['seasonal_anomaly']]
                    ax.scatter(seasonal_anomalies['date'], seasonal_anomalies['value'], 
                              color='orange', marker='d', s=50, label='Seasonal anomalies')
                
                if show_severity and result['is_anomaly'].any():
                    anomaly_points = result[result['is_anomaly']]
                    for i, row in anomaly_points.iterrows():
                        severity = row['anomaly_severity']
                        ax.annotate(f"{severity}", 
                                   (row['date'], row['value']),
                                   xytext=(5, 5),
                                   textcoords='offset points',
                                   fontsize=9,
                                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
                
                ax.set_title(f'Anomaly Detection for {col}', fontsize=14)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel(col, fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                fig.autofmt_xdate()
                plt.tight_layout()
                plt.show()
    

    def plot_seasonal_decomposition(self, column, figsize=(15, 10)):
        """
        Plot the seasonal decomposition for a specific column.
        
        Parameters:
        -----------
        column : str
            Column name to analyze
        figsize : tuple
            Figure size
        """
        if column not in self.value_cols:
            print(f"Column {column} not found in value columns.")
            return

        temp_df = self.df_processed.set_index(self.date_col)[[column]]
        if len(temp_df) < 14:
            period = 2
        elif len(temp_df) < 60:
            period = 7
        else:
            period = 30
        
        try:
            decomposition = seasonal_decompose(
                temp_df, 
                model='additive', 
                period=period,
                extrapolate_trend=True
            )
            
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            axes[0].plot(decomposition.observed)
            axes[0].set_title('Observed', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(decomposition.trend)
            axes[1].set_title('Trend', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(decomposition.seasonal)
            axes[2].set_title(f'Seasonality (Period={period})', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[3].plot(decomposition.resid)
            axes[3].set_title('Residuals', fontsize=12)
            axes[3].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in seasonal decomposition: {e}")
    

    def plot_distribution(self, column, figsize=(15, 5)):
        """
        Plot the distribution of values for a column, highlighting anomalies.
        
        Parameters:
        -----------
        column : str
            Column name to analyze
        figsize : tuple
            Figure size
        """
        if column not in self.results:
            print(f"Column {column} not found in results.")
            return
            
        result = self.results[column]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        sns.histplot(result['value'], kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {column}', fontsize=12)
        anomalies = result[result['is_anomaly']]['value']
        if not anomalies.empty:
            ax1.axvline(anomalies.min(), color='r', linestyle='--', alpha=0.7)
            ax1.axvline(anomalies.max(), color='r', linestyle='--', alpha=0.7)
        
        sns.boxplot(y=result['value'], ax=ax2)
        ax2.set_title(f'Box Plot of {column}', fontsize=12)
        if not anomalies.empty:
            y_pos = [0] * len(anomalies)
            ax2.scatter(y_pos, anomalies, color='red', s=50, label='Anomalies')
        plt.tight_layout()
        plt.show()

    
    def get_anomaly_summary(self):
        """
        Get a summary of detected anomalies across all methods and columns.
        
        Returns:
        --------
        pandas.DataFrame: Summary statistics of anomalies by column and method
        """
        summary = []
        
        for col in self.value_cols:
            if col in self.results:
                result = self.results[col]
                summary.append({
                    'column': col,
                    'total_points': len(result),
                    'zscore_anomalies': sum(result['zscore_anomaly']) if 'zscore_anomaly' in result else 0,
                    'iqr_anomalies': sum(result['iqr_anomaly']) if 'iqr_anomaly' in result else 0,
                    'iforest_anomalies': sum(result['iforest_anomaly']) if 'iforest_anomaly' in result else 0,
                    'seasonal_anomalies': sum(result['seasonal_anomaly']) if 'seasonal_anomaly' in result else 0,
                    'total_anomalies': sum(result['is_anomaly']),
                    'anomaly_percentage': sum(result['is_anomaly']) / len(result) * 100,
                    'max_severity': result['anomaly_severity'].max() if sum(result['is_anomaly']) > 0 else 0,
                    'avg_severity': result['anomaly_severity'][result['is_anomaly']].mean() if sum(result['is_anomaly']) > 0 else 0,
                    'skew': result['skew'].iloc[0],
                    'kurtosis': result['kurtosis'].iloc[0]
                })
        return pd.DataFrame(summary)