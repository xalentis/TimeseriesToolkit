import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
from sklearn.preprocessing import LabelEncoder


class TimeSeriesSuitabilityScorer:
    """Evaluates whether a DataFrame is suitable for time series modeling."""
    
    def __init__(self, min_observations: int = 10):
        self.min_observations = min_observations
        self.encoders = {}

    
    def encode_categorical_columns(self, df: pd.DataFrame, encoding_type: str = 'label',  drop_original: bool = True) -> Tuple[pd.DataFrame, Dict]:

        df_encoded = df.copy()
        encoders = {}
        categorical_columns = [col for col in df.columns 
                             if col.lower() != 'date' and 
                             (df[col].dtype == 'object' or df[col].dtype.name == 'category')]
        
        if encoding_type in ['label', 'both']:
            for col in categorical_columns:
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[f'{col}_label'] = le
        if encoding_type in ['onehot', 'both']:
            for col in categorical_columns:
                dummies = pd.get_dummies(df[col], prefix=f'{col}')
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                encoders[f'{col}_onehot'] = list(dummies.columns)
        if drop_original:
            df_encoded = df_encoded.drop(columns=categorical_columns)
        
        return df_encoded, encoders
    

    def score(self, df: pd.DataFrame, date_column: str = None, value_columns: List[str] = None) -> Dict[str, Any]:
        results = {
            'overall_score': 0.0,
            'is_timeseries_suitable': False,
            'detailed_scores': {},
            'recommendations': [],
            'identified_datetime_column': None,
            'identified_value_columns': [],
            'issues': []
        }
        
        if df.empty:
            results['issues'].append("DataFrame is empty")
            return results

        df_encoded, self.encoders = self.encode_categorical_columns(df)
        
        # score components
        datetime_score, detected_date_col = self._score_datetime_column(df_encoded, date_column)
        value_score, detected_value_cols = self._score_value_columns(df_encoded, value_columns, detected_date_col)
        quantity_score = self._score_data_quantity(df_encoded)
        temporal_score = self._score_temporal_properties(df_encoded, detected_date_col)
        quality_score = self._score_data_quality(df_encoded, detected_value_cols)
        characteristics_score = self._score_timeseries_characteristics(df_encoded, detected_date_col, detected_value_cols)
        
        results['detailed_scores'] = {
            'datetime_detection': datetime_score,
            'value_columns': value_score,
            'data_quantity': quantity_score,
            'temporal_properties': temporal_score,
            'data_quality': quality_score,
            'timeseries_characteristics': characteristics_score
        }
        results['overall_score'] = sum(results['detailed_scores'].values())
        results['is_timeseries_suitable'] = results['overall_score'] >= 60
        results['identified_datetime_column'] = detected_date_col
        results['identified_value_columns'] = detected_value_cols
        results['recommendations'], results['issues'] = self._generate_recommendations_and_issues(results['detailed_scores'])
        return results
    

    def _score_datetime_column(self, df: pd.DataFrame, date_column: str = None) -> Tuple[float, str]:
        if date_column and date_column in df.columns:
            try:
                pd.to_datetime(df[date_column])
                return 25.0, date_column
            except:
                pass
        
        # detect datetime columns
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'created', 'updated']):
                try:
                    pd.to_datetime(df[col])
                    return 20.0, col
                except:
                    continue
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.8:
                    return 15.0, col
            except:
                continue
        
        if isinstance(df.index, pd.DatetimeIndex):
            return 25.0, df.index.name or 'index'
        try:
            pd.to_datetime(df.index)
            return 20.0, 'index'
        except:
            pass
        return 0.0, None
    

    def _score_value_columns(self, df: pd.DataFrame, value_columns: List[str] = None, 
                           date_column: str = None) -> Tuple[float, List[str]]:
        if value_columns:
            valid_cols = [col for col in value_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            if valid_cols:
                return 15.0, valid_cols
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if date_column and date_column in numeric_cols:
            numeric_cols.remove(date_column)
        if len(numeric_cols) >= 1:
            return 15.0 if len(numeric_cols) >= 2 else 12.0, numeric_cols
        return 0.0, []
    

    def _score_data_quantity(self, df: pd.DataFrame) -> float:
        n_obs = len(df)
        if n_obs < self.min_observations:
            return 0.0
        elif n_obs < 30:
            return 8.0
        elif n_obs < 100:
            return 12.0
        else:
            return 15.0
    

    def _score_temporal_properties(self, df: pd.DataFrame, date_column: str) -> float:
        if not date_column:
            return 0.0
        
        score = 0.0
        try:
            dates = pd.to_datetime(df.index if date_column == 'index' else df[date_column])
            # check ordering
            if dates.is_monotonic_increasing:
                score += 8.0
            elif dates.is_monotonic_decreasing:
                score += 6.0
            else:
                score += 2.0
            
            # check regularity
            if len(dates) > 2:
                intervals = dates.diff().dropna()
                unique_intervals = len(intervals.unique())
                if unique_intervals == 1:
                    score += 8.0
                elif unique_intervals <= 3:
                    score += 6.0
                else:
                    most_common = intervals.mode().iloc[0] if not intervals.mode().empty else None
                    if most_common and (intervals == most_common).sum() / len(intervals) > 0.8:
                        score += 4.0
                    else:
                        score += 2.0
            
            # time span
            time_span = dates.max() - dates.min()
            if time_span > pd.Timedelta(days=30):
                score += 4.0
            elif time_span > pd.Timedelta(days=7):
                score += 2.0
                
        except Exception:
            pass
        
        return score
    

    def _score_data_quality(self, df: pd.DataFrame, value_columns: List[str]) -> float:
        if not value_columns:
            return 0.0
        
        score = 0.0
        
        # missing values
        missing_ratio = df[value_columns].isnull().sum().sum() / (len(df) * len(value_columns))
        if missing_ratio < 0.05:
            score += 8.0
        elif missing_ratio < 0.15:
            score += 6.0
        elif missing_ratio < 0.30:
            score += 4.0
        else:
            score += 2.0
        
        # variance
        for col in value_columns:
            if df[col].nunique() > 1:
                score += min(2.0, 7.0 / len(value_columns))
        return min(score, 15.0)
    

    def _score_timeseries_characteristics(self, df: pd.DataFrame, date_column: str, value_columns: List[str]) -> float:
        if not date_column or not value_columns:
            return 0.0
        
        score = 0.0
        try:
            for col in value_columns[:2]:
                values = df[col].dropna()
                if len(values) > 10:
                    # trend detection
                    x = np.arange(len(values))
                    correlation = np.corrcoef(x, values)[0, 1]
                    if abs(correlation) > 0.3:
                        score += 2.0
                    
                    # seasonality check
                    if len(values) > 24:
                        try:
                            rolling_mean = values.rolling(window=min(12, len(values)//4)).mean()
                            if rolling_mean.std() > 0:
                                score += 1.0
                        except:
                            pass
            
            if len(value_columns) > 1:
                score += 2.0
        except Exception:
            pass
        return min(score, 10.0)
    

    def _generate_recommendations_and_issues(self, scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        recommendations = []
        issues = []
        if scores['datetime_detection'] < 15:
            issues.append("No clear datetime column identified")
            recommendations.append("Ensure you have a column with datetime values")
        if scores['value_columns'] < 10:
            issues.append("Insufficient numeric columns for time series analysis")
            recommendations.append("Include numeric columns representing values over time")
        if scores['data_quantity'] < 10:
            issues.append(f"Insufficient data points (minimum: {self.min_observations})")
            recommendations.append("Collect more data points for meaningful analysis")
        if scores['temporal_properties'] < 10:
            issues.append("Poor temporal structure")
            recommendations.append("Sort data by datetime and ensure regular intervals")
        if scores['data_quality'] < 10:
            issues.append("Data quality issues")
            recommendations.append("Handle missing values and ensure meaningful variation")
        if scores['timeseries_characteristics'] < 5:
            issues.append("Data lacks time series characteristics")
            recommendations.append("Verify data represents measurements changing over time")
        return recommendations, issues