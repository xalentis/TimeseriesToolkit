import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesFeatureEngineering:
    
    def __init__(self, 
                 correlation_threshold: float = 0.95,
                 importance_threshold: float = 0.01,
                 categorical_threshold: int = 10):
        """
        Initialize the feature engineering framework.
        
        Parameters:
        -----------
        correlation_threshold : float, default=0.95
            Threshold for removing highly correlated features
        importance_threshold : float, default=0.01
            Minimum importance score for feature selection
        categorical_threshold : int, default=10
            Maximum unique values for categorical detection
        """
        self.scalers = {}
        self.label_encoders = {}
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
        self.categorical_threshold = categorical_threshold
        self.date_col = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.target_col = None
        self.feature_importance_scores = {}
        self.selected_features = []
        self.metadata = {
            'original_columns': [],
            'created_features': [],
            'column_dtypes': {},
            'seasonal_info': {}
        }

    
    def identify_and_convert_columns(self, df: pd.DataFrame, 
                                   date_columns: Optional[List[str]] = None,
                                   target_column: Optional[str] = None,
                                   verbose: bool = False) -> pd.DataFrame:
        """
        Identify and convert column types automatically.
        """
        df_processed = df.copy()
        self.target_col = target_column
        self.metadata['original_columns'] = list(df.columns)
        
        # store original dtypes
        for col in df.columns:
            self.metadata['column_dtypes'][col] = str(df[col].dtype)
        
        # detect date columns
        detected_date_cols = []
        if date_columns is None:
            date_columns = []
            # pattern-based detection
            date_pattern = re.compile(r'date|time|day|month|year|dt_', re.IGNORECASE)
            for col in df.columns:
                if date_pattern.search(col):
                    detected_date_cols.append(col)
            
            # try parsing object columns as dates
            for col in df.select_dtypes(include=['object']).columns:
                if col not in detected_date_cols and df[col].nunique() < 1000:
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        detected_date_cols.append(col)
                    except:
                        pass
            
            date_columns.extend(detected_date_cols)
            date_columns = list(set(date_columns))
        
        # convert date columns
        for col in date_columns:
            if col in df_processed.columns:
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    if verbose:
                        print(f"Converted {col} to datetime")
                except:
                    if verbose:
                        print(f"Failed to convert {col} to datetime")
        
        # identify column types
        date_cols = []
        numeric_cols = []
        categorical_cols = []
        
        for col in df_processed.columns:
            if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                date_cols.append(col)
            elif df_processed[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                if df_processed[col].nunique() > self.categorical_threshold:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            elif df_processed[col].dtype == 'object':
                if col not in date_columns:
                    n_unique = df_processed[col].nunique()
                    if n_unique <= self.categorical_threshold:
                        categorical_cols.append(col)
                    else:
                        try:
                            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                            numeric_cols.append(col)
                            if verbose:
                                print(f"Converted {col} to numeric")
                        except:
                            categorical_cols.append(col)
        
        # store column classifications
        self.date_col = date_cols[0] if date_cols else None
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        
        # encode categorical variables
        for col in self.categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                if verbose:
                    print(f"Encoded categorical column {col}")
        
        return df_processed
    

    def detect_lag_lead_periods(self, df: pd.DataFrame, 
                               target_column: str,
                               date_column: str,
                               min_lag: int = 1,
                               max_lag: int = 30,
                               min_lead: int = 1,
                               max_lead: int = 15,
                               max_periods: int = 5,
                               correlation_threshold: float = 0.2,
                               verbose: bool = False) -> Tuple[List[int], List[int]]:
        """
        Automatically detect optimal lag and lead periods.
        """
        df_copy = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        df_copy = df_copy.dropna(subset=[target_column, date_column])
        df_copy = df_copy.sort_values(by=date_column)
        
        if not pd.api.types.is_numeric_dtype(df_copy[target_column]):
            df_copy[target_column] = pd.to_numeric(df_copy[target_column], errors='coerce')
            df_copy = df_copy.dropna(subset=[target_column])
        
        df_ts = df_copy.set_index(date_column)
        series = df_ts[target_column]
        
        lag_periods = []
        lead_periods = []
        
        # seasonality detection
        seasonal_periods = []
        try:
            if len(series) >= 24:
                decomposition = seasonal_decompose(series, model='additive', period=min(len(series)//2, 365))
                seasonal = decomposition.seasonal 
                seasonal_diff = np.diff(seasonal)
                sign_changes = np.where(np.diff(np.signbit(seasonal_diff)))[0]
                if len(sign_changes) > 1:
                    peak_distances = np.diff(sign_changes)
                    unique_distances, counts = np.unique(peak_distances, return_counts=True)
                    sorted_indices = np.argsort(-counts)
                    for idx in sorted_indices[:3]:
                        period = unique_distances[idx]
                        if period >= min_lag and period <= max_lag:
                            seasonal_periods.append(int(period))
        except:
            pass
        
        # acf for lag identification
        try:
            acf_values = acf(series, nlags=max_lag, fft=True)
            significant_lags = []
            
            for i in range(min_lag, min(len(acf_values), max_lag + 1)):
                if abs(acf_values[i]) > correlation_threshold:
                    significant_lags.append(i)
            
            # add seasonal periods
            for period in seasonal_periods:
                if period not in significant_lags:
                    significant_lags.append(period)
            
            # sort by correlation strength
            lag_corr = [(lag, abs(acf_values[lag])) for lag in significant_lags 
                       if lag < len(acf_values)]
            lag_corr.sort(key=lambda x: x[1], reverse=True)
            lag_periods = [lag for lag, _ in lag_corr[:max_periods]]
            
        except:
            lag_periods = seasonal_periods[:max_periods]
        
        # cross-correlation for leads
        try:
            lead_corr = []
            for lead in range(min_lead, max_lead + 1):
                shifted_series = series.shift(-lead)
                corr = series.corr(shifted_series)
                if not pd.isna(corr) and abs(corr) > correlation_threshold:
                    lead_corr.append((lead, abs(corr)))
            
            lead_corr.sort(key=lambda x: x[1], reverse=True)
            lead_periods = [lead for lead, _ in lead_corr[:max_periods]]
        except:
            lead_periods = list(range(min_lead, min(max_lead, 4)))[:max_periods]
        
        # defaults if nothing found
        if not lag_periods:
            lag_periods = [1, 7, 14] if max_lag >= 14 else [1, 2, 3]
        if not lead_periods:
            lead_periods = [1, 2, 3][:max_periods]
        
        # ensure within bounds
        lag_periods = [int(lag) for lag in lag_periods if min_lag <= lag <= max_lag]
        lead_periods = [int(lead) for lead in lead_periods if min_lead <= lead <= max_lead]
        
        lag_periods.sort()
        lead_periods.sort()
        
        if verbose:
            print(f"Detected lag periods: {lag_periods}")
            print(f"Detected lead periods: {lead_periods}")
        
        return lag_periods, lead_periods
    

    def create_time_series_features(self, df: pd.DataFrame,
                                  make_stationary: bool = False,
                                  remove_seasonal: bool = False) -> pd.DataFrame:
        """
        Create time series specific features (stationarity, seasonality removal).
        """
        df_processed = df.copy()
        
        # stationarity
        if make_stationary and len(df_processed) >= 3:
            cols_to_process = [col for col in self.numeric_cols if col != self.target_col]
            
            for col in cols_to_process:
                if col in df_processed.columns and len(df_processed[col].dropna()) > 2:
                    try:
                        result = adfuller(df_processed[col].dropna())
                        if result[1] > 0.05:
                            df_processed[col] = df_processed[col].diff()
                            
                            if len(df_processed[col].dropna()) > 0:
                                result = adfuller(df_processed[col].dropna())
                                if result[1] > 0.05 and len(df_processed[col].dropna()) > 1:
                                    df_processed[col] = df_processed[col].diff()
                    except:
                        continue
            
            df_processed = df_processed.dropna()
        
        # seasonality removal
        if remove_seasonal and len(df_processed) >= 24:
            cols_to_process = [col for col in self.numeric_cols if col != self.target_col]
            
            for col in cols_to_process:
                if col in df_processed.columns and len(df_processed[col].dropna()) >= 24:
                    try:
                        periods = [12, 7, 4, 24]
                        best_period = 12
                        
                        for period in periods:
                            if len(df_processed[col].dropna()) >= 2 * period:
                                best_period = period
                                break
                        
                        decomposition = seasonal_decompose(
                            df_processed[col].dropna(), 
                            model='additive', 
                            period=best_period
                        )
                        
                        if len(decomposition.resid.dropna()) > 0:
                            df_processed[col] = decomposition.resid
                    except:
                        continue
            
            df_processed = df_processed.dropna()
        
        return df_processed
    

    def create_advanced_features(self, df: pd.DataFrame,
                               lag_periods: List[int] = [1, 3, 7, 14],
                               lead_periods: List[int] = [1, 3],
                               verbose: bool = False) -> pd.DataFrame:
        """
        Create comprehensive feature set including lags, leads, rolling stats, and technical indicators.
        """
        df_feat = df.copy()
        
        # sort by date if date column exists
        if self.date_col and self.date_col in df_feat.columns:
            df_feat = df_feat.sort_values(self.date_col)
        
        # only engineer features from non-target numeric columns to avoid data leakage
        cols_to_engineer = [col for col in self.numeric_cols if col != self.target_col]
        
        # create lag/lead features for target column
        if self.target_col and self.target_col in df_feat.columns:
            for lag in lag_periods:
                if lag < len(df_feat):
                    feature_name = f'{self.target_col}_lag_{lag}'
                    df_feat[feature_name] = df_feat[self.target_col].shift(lag)
                    self.metadata['created_features'].append(feature_name)
            
            for lead in lead_periods:
                if lead < len(df_feat):
                    feature_name = f'{self.target_col}_lead_{lead}'
                    df_feat[feature_name] = df_feat[self.target_col].shift(-lead)
                    self.metadata['created_features'].append(feature_name)
        
        # features for other numeric columns
        windows = [3, 7, 14, 30]
        
        for col in cols_to_engineer:
            if col not in df_feat.columns or len(df_feat[col].dropna()) < 3:
                continue
            
            # rolling statistics
            for window in windows:
                if window < len(df_feat):
                    rolling = df_feat[col].rolling(window=window, min_periods=1)
                    
                    features = {
                        f'{col}_roll_mean_{window}': rolling.mean(),
                        f'{col}_roll_std_{window}': rolling.std(),
                        f'{col}_roll_min_{window}': rolling.min(),
                        f'{col}_roll_max_{window}': rolling.max(),
                        f'{col}_roll_median_{window}': rolling.median()
                    }
                    
                    for feat_name, feat_values in features.items():
                        df_feat[feat_name] = feat_values
                        self.metadata['created_features'].append(feat_name)
            

            # exponential moving averages
            alphas = [0.1, 0.3, 0.5]
            ema_features = {}
            created_feature_names = []

            for alpha in alphas:
                feature_name = f'{col}_ema_{alpha}'
                ema_features[feature_name] = df_feat[col].ewm(alpha=alpha).mean()
                created_feature_names.append(feature_name)

            # Batch assign all EMA features at once
            if ema_features:
                df_feat = df_feat.assign(**ema_features)
                self.metadata['created_features'].extend(created_feature_names)
            
            # rate of change and momentum
            for period in [1, 3, 7]:
                if period < len(df_feat):
                    roc_name = f'{col}_roc_{period}'
                    momentum_name = f'{col}_momentum_{period}'
                    df_feat[roc_name] = df_feat[col].pct_change(periods=period)
                    df_feat[momentum_name] = df_feat[col] - df_feat[col].shift(period)
                    self.metadata['created_features'].extend([roc_name, momentum_name])
            
            # volatility measures
            for window in [7, 14, 30]:
                if window < len(df_feat):
                    returns = df_feat[col].pct_change()
                    feature_name = f'{col}_volatility_{window}'
                    df_feat[feature_name] = returns.rolling(window).std()
                    self.metadata['created_features'].append(feature_name)
        
        # date/time features
        if self.date_col and self.date_col in df_feat.columns:
            date_series = pd.to_datetime(df_feat[self.date_col])
            
            date_features = {
                'year': date_series.dt.year,
                'month': date_series.dt.month,
                'day': date_series.dt.day,
                'dayofweek': date_series.dt.dayofweek,
                'dayofyear': date_series.dt.dayofyear,
                'quarter': date_series.dt.quarter,
                'is_weekend': (date_series.dt.dayofweek >= 5).astype(int),
                'is_month_start': date_series.dt.is_month_start.astype(int),
                'is_month_end': date_series.dt.is_month_end.astype(int)
            }
            
            for feat_name, feat_values in date_features.items():
                df_feat[feat_name] = feat_values
                self.metadata['created_features'].append(feat_name)
        
        # interaction features
        numeric_features = [col for col in df_feat.columns if col in cols_to_engineer]
        if len(numeric_features) >= 2:
            # Pre-allocate arrays to store new features
            new_features = {}
            created_feature_names = []
            
            for i, col1 in enumerate(numeric_features[:3]):
                for col2 in numeric_features[i+1:4]:
                    if col1 != col2:
                        ratio_name = f'{col1}_{col2}_ratio'
                        diff_name = f'{col1}_{col2}_diff'
                        
                        # Calculate features and store in dictionary
                        new_features[ratio_name] = df_feat[col1] / (df_feat[col2] + 1e-8)
                        new_features[diff_name] = df_feat[col1] - df_feat[col2]
                        
                        created_feature_names.extend([ratio_name, diff_name])
            
            # Batch assign all new features at once
            if new_features:
                df_feat = df_feat.assign(**new_features)
                self.metadata['created_features'].extend(created_feature_names)
        
        # remove infinite values
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        
        if verbose:
            print(f"Created {len(self.metadata['created_features'])} new features")
        
        return df_feat
    

    def create_seasonal_indicators(self, df: pd.DataFrame,
                                 seasonal_periods: List[int]) -> pd.DataFrame:
        """
        Create seasonal indicator features.
        """
        df_processed = df.copy()
        
        if not self.target_col or self.target_col not in df_processed.columns:
            return df_processed
        
        for period in seasonal_periods:
            # seasonal rolling statistics
            seasonal_mean = df_processed[self.target_col].rolling(window=period, center=False).mean()
            seasonal_std = df_processed[self.target_col].rolling(window=period, center=False).std()
            
            # indicators
            features = {
                f'above_seasonal_mean_{period}': (df_processed[self.target_col] > seasonal_mean).astype(int),
                f'below_seasonal_mean_{period}': (df_processed[self.target_col] < seasonal_mean).astype(int),
                f'seasonal_zscore_{period}': (df_processed[self.target_col] - seasonal_mean) / seasonal_std.replace(0, 1),
                f'seasonal_rise_{period}': (df_processed[self.target_col] > df_processed[self.target_col].shift(period)).astype(int),
                f'seasonal_fall_{period}': (df_processed[self.target_col] < df_processed[self.target_col].shift(period)).astype(int)
            }
            
            for feat_name, feat_values in features.items():
                df_processed[feat_name] = feat_values
                self.metadata['created_features'].append(feat_name)
        
        # fill nan values
        for col in df_processed.columns:
            if col.startswith(('above_seasonal', 'below_seasonal', 'seasonal_rise', 'seasonal_fall')):
                df_processed[col] = df_processed[col].fillna(0)
            elif col.startswith('seasonal_zscore'):
                df_processed[col] = df_processed[col].fillna(0)
        
        return df_processed
    

    def handle_missing_values(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Intelligent missing value handling.
        """
        df_clean = df.copy()
        
        # forward fill then backward fill
        df_clean = df_clean.ffill().bfill()
        
        # handle remaining NaNs
        for col in df_clean.columns:
            if df_clean[col].isna().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    mode_val = df_clean[col].mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else 0
                    df_clean[col] = df_clean[col].fillna(fill_val)
                
                if verbose:
                    print(f"Filled missing values in {col}")
        
        return df_clean.dropna(how='all')
    

    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Calculate feature importance using multiple methods.
        """
        if target_col not in df.columns or len(df) < 5:
            return {}
        
        feature_cols = [col for col in df.columns 
                       if col != target_col and col != self.date_col]
        
        if len(feature_cols) == 0:
            return {}
        
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        importance_scores = {}
        
        try:
            # correlation-based importance
            correlations = abs(df[feature_cols + [target_col]].corr()[target_col].drop(target_col))
            for col, corr in correlations.items():
                importance_scores[col] = importance_scores.get(col, 0) + corr * 0.3
            
            # mutual information
            if len(X) > 10:
                mi_scores = mutual_info_regression(X, y, random_state=42)
                for i, col in enumerate(feature_cols):
                    importance_scores[col] = importance_scores.get(col, 0) + mi_scores[i] * 0.3
            
            # random Forest importance
            if len(X) > 10 and len(feature_cols) > 1:
                rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                rf.fit(X, y)
                for i, col in enumerate(feature_cols):
                    importance_scores[col] = importance_scores.get(col, 0) + rf.feature_importances_[i] * 0.4
        
        except Exception as e:
            # fallback to correlation only
            try:
                correlations = abs(df[feature_cols + [target_col]].corr()[target_col].drop(target_col))
                importance_scores = correlations.to_dict()
            except:
                importance_scores = {col: 0.1 for col in feature_cols}
        
        return importance_scores
    

    def remove_correlated_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Remove highly correlated features.
        """
        if target_col:
            feature_cols = [col for col in df.columns 
                           if col != target_col and col != self.date_col]
        else:
            feature_cols = [col for col in df.columns if col != self.date_col]
        
        if len(feature_cols) <= 1:
            return df
        
        try:
            corr_matrix = abs(df[feature_cols].corr())
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = set()
            
            for col in upper_triangle.columns:
                correlated_features = upper_triangle[col][
                    upper_triangle[col] > self.correlation_threshold
                ].index.tolist()
                
                if correlated_features:
                    if target_col and target_col in df.columns:
                        target_corrs = abs(df[[col] + correlated_features + [target_col]].corr()[target_col])
                        keep_feature = target_corrs.drop(target_col).idxmax()
                        to_drop.update([f for f in [col] + correlated_features if f != keep_feature])
                    else:
                        to_drop.update(correlated_features)
            
            features_to_keep = [col for col in df.columns if col not in to_drop]
            return df[features_to_keep]
        
        except:
            return df
    

    def select_important_features(self, df: pd.DataFrame, 
                                target_col: str, 
                                max_features: int = 50) -> pd.DataFrame:
        """
        Select most important features based on multiple criteria.
        """
        if target_col not in df.columns:
            return df
        
        importance_scores = self.calculate_feature_importance(df, target_col)
        self.feature_importance_scores = importance_scores
        
        if not importance_scores:
            return df
        
        # sort features by importance
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # select top features
        important_features = [
            feat for feat, score in sorted_features 
            if score >= self.importance_threshold
        ][:max_features]
        
        # always keep target and date columns
        columns_to_keep = important_features + [target_col]
        if self.date_col and self.date_col in df.columns:
            columns_to_keep.append(self.date_col)
        
        self.selected_features = important_features
        return df[columns_to_keep]
    

    def scale_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Scale numeric features excluding target and date columns.
        """
        df_scaled = df.copy()
        
        # identify columns to scale
        numeric_cols_to_scale = []
        for col in df_scaled.columns:
            if col != target_col and col != self.date_col:
                if df_scaled[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    if not df_scaled[col].empty and df_scaled[col].var() > 0:
                        numeric_cols_to_scale.append(col)
        
        # scale features
        for col in numeric_cols_to_scale:
            if len(df_scaled[col].dropna()) > 0:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]])
                else:
                    df_scaled[col] = self.scalers[col].transform(df_scaled[[col]])
        
        return df_scaled
    

    def engineer_features(self, df: pd.DataFrame,
                        target_column: Optional[str] = None,
                        date_columns: Optional[List[str]] = None,
                        auto_detect_lags: bool = True,
                        lag_periods: Optional[List[int]] = None,
                        lead_periods: Optional[List[int]] = None,
                        make_stationary: bool = False,
                        remove_seasonal: bool = False,
                        create_seasonal_indicators: bool = True,
                        feature_selection: bool = True,
                        scale_features: bool = True,
                        max_features: int = 50,
                        verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main method that orchestrates the entire feature engineering pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_column : str, optional
            Target variable column name
        date_columns : List[str], optional
            Date column names (auto-detected if None)
        auto_detect_lags : bool, default=True
            Whether to automatically detect optimal lag/lead periods
        lag_periods : List[int], optional
            Manual lag periods (used if auto_detect_lags=False)
        lead_periods : List[int], optional
            Manual lead periods (used if auto_detect_lags=False)
        make_stationary : bool, default=False
            Whether to make time series stationary
        remove_seasonal : bool, default=False
            Whether to remove seasonal components
        create_seasonal_indicators : bool, default=True
            Whether to create seasonal indicator features
        feature_selection : bool, default=True
            Whether to perform feature selection
        scale_features : bool, default=True
            Whether to scale features
        max_features : int, default=50
            Maximum number of features to select
        verbose : bool, default=False
            Whether to print processing information
        
        Returns:
        --------
        processed_df : pd.DataFrame
            Dataframe with engineered features
        metadata : Dict[str, Any]
            Processing metadata and information
        """
        
        if verbose:
            print("Starting comprehensive feature engineering pipeline...")
        
        # 1: identify and convert column types
        processed_df = self.identify_and_convert_columns(
            df, date_columns, target_column, verbose
        )
        
        # 2: auto-detect lag/lead periods if requested
        if auto_detect_lags and self.date_col and target_column:
            detected_lags, detected_leads = self.detect_lag_lead_periods(
                processed_df, target_column, self.date_col, verbose=verbose
            )
            if not lag_periods:
                lag_periods = detected_lags
            if not lead_periods:
                lead_periods = detected_leads
        else:
            lag_periods = lag_periods or [1, 3, 7]
            lead_periods = lead_periods or [1, 3]
        
        if verbose:
            print(f"Using lag periods: {lag_periods}")
            print(f"Using lead periods: {lead_periods}")
        
        # 3: handle missing values
        processed_df = self.handle_missing_values(processed_df, verbose)
        
        # 4: create time series features (stationarity/seasonality)
        processed_df = self.create_time_series_features(
            processed_df, make_stationary, remove_seasonal
        )
        
        # 5: create advanced features
        processed_df = self.create_advanced_features(
            processed_df, lag_periods, lead_periods, verbose
        )
        
        # 6: create seasonal indicators
        if create_seasonal_indicators and lag_periods:
            seasonal_periods = [p for p in lag_periods if p >= 7]  # use longer periods for seasonality
            if seasonal_periods:
                processed_df = self.create_seasonal_indicators(
                    processed_df, seasonal_periods
                )
        
        # 7: handle missing values again after feature creation
        processed_df = self.handle_missing_values(processed_df, verbose)
        
        # 8: remove highly correlated features
        processed_df = self.remove_correlated_features(processed_df, target_column)
        
        # 9: feature selection
        if feature_selection and target_column:
            processed_df = self.select_important_features(
                processed_df, target_column, max_features
            )
        
        # 10: scale features
        if scale_features:
            processed_df = self.scale_features(processed_df, target_column)
        
        # update metadata
        self.metadata.update({
            'final_shape': processed_df.shape,
            'final_columns': list(processed_df.columns),
            'lag_periods_used': lag_periods,
            'lead_periods_used': lead_periods,
            'feature_importance_scores': self.feature_importance_scores,
            'selected_features': self.selected_features,
            'processing_steps': {
                'column_identification': True,
                'missing_value_handling': True,
                'stationarity_applied': make_stationary,
                'seasonality_removed': remove_seasonal,
                'seasonal_indicators_created': create_seasonal_indicators,
                'correlation_removal': True,
                'feature_selection': feature_selection,
                'scaling_applied': scale_features
            }
        })
        
        if verbose:
            print(f"Feature engineering completed!")
            print(f"Original shape: {df.shape}")
            print(f"Final shape: {processed_df.shape}")
            print(f"Created {len(self.metadata['created_features'])} new features")
            if self.selected_features:
                print(f"Selected {len(self.selected_features)} important features")
        
        return processed_df, self.metadata
    

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature engineering process.
        """
        summary = {
            'original_columns': self.metadata.get('original_columns', []),
            'created_features': self.metadata.get('created_features', []),
            'selected_features': self.selected_features,
            'feature_importance_scores': self.feature_importance_scores,
            'column_types': {
                'date_column': self.date_col,
                'numeric_columns': self.numeric_cols,
                'categorical_columns': self.categorical_cols,
                'target_column': self.target_col
            },
            'processing_parameters': {
                'correlation_threshold': self.correlation_threshold,
                'importance_threshold': self.importance_threshold,
                'categorical_threshold': self.categorical_threshold
            },
            'final_shape': self.metadata.get('final_shape'),
            'processing_steps': self.metadata.get('processing_steps', {})
        }
        
        return summary
    
