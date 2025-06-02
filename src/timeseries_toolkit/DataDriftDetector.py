import pandas as pd
import numpy as np
from scipy import stats
from typing import Any, Dict, List, Optional, Union
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gaussian_kde


class DataDriftDetector:
    """
    A comprehensive data drift detection class that identifies and quantifies
    drift between two datasets using multiple statistical methods.
    
    This class provides methods to detect various types of drift including:
    - Feature-level drift (categorical and numerical)
    - Distribution drift
    - Statistical drift
    - Dimensional drift
    - Outlier pattern drift
    - Correlation structure drift
    """
    
    def __init__(self, 
                 drift_threshold: float = 0.7,
                 categorical_threshold: int = 10,
                 random_state: int = 42):
        """
        Initialize the DataDriftDetector.
        
        Parameters:
        -----------
        drift_threshold : float, default=0.7
            Threshold above which drift is considered significant
        categorical_threshold : int, default=10
            Maximum unique values for a column to be considered categorical
        random_state : int, default=42
            Random state for reproducible results
        """
        self.drift_threshold = drift_threshold
        self.categorical_threshold = categorical_threshold
        self.random_state = random_state
        self.results_ = None
        self.reference_data_ = None
        self.comparison_data_ = None
        

    def detect_drift(self,
                    reference_data: pd.DataFrame,
                    comparison_data: pd.DataFrame,
                    categorical_columns: Optional[List[str]] = None,
                    date_columns: Optional[List[str]] = None,
                    importance_weights: Optional[Dict[str, float]] = None,
                    verbose: bool = False) -> Dict[str, Union[float, Dict]]:
        """
        Detect and quantify data drift between two dataframes.
        
        Parameters:
        -----------
        reference_data : pd.DataFrame
            Reference dataframe (usually training data)
        comparison_data : pd.DataFrame
            Comparison dataframe (usually new/production data)
        categorical_columns : List[str], optional
            List of categorical column names (auto-detected if None)
        date_columns : List[str], optional
            List of date column names to exclude from analysis
        importance_weights : Dict[str, float], optional
            Dictionary mapping feature names to their importance weights
        verbose : bool, default=False
            Whether to print detailed drift analysis information
        
        Returns:
        --------
        Dict[str, Union[float, Dict]]
            Dictionary containing drift scores, details and recommendations
        """
        # Store data for potential reuse
        self.reference_data_ = reference_data.copy()
        self.comparison_data_ = comparison_data.copy()
        
        # Validate inputs and prepare data
        common_columns = self._validate_and_prepare_data(
            reference_data, comparison_data, date_columns
        )
        
        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = self._detect_categorical_columns(
                reference_data, comparison_data, common_columns
            )
        
        # Filter out date columns from categorical_columns if they exist
        if date_columns:
            categorical_columns = [col for col in categorical_columns 
                                 if col not in date_columns]
        
        numerical_columns = [col for col in common_columns 
                           if col not in categorical_columns]
        
        # Filter numerical columns to ensure they can be converted to numeric
        numerical_columns = self._validate_numerical_columns(
            reference_data, comparison_data, numerical_columns, verbose
        )
        
        # Initialize results dictionary
        results = self._initialize_results()
        
        # Perform drift detection
        results = self._detect_feature_drift(
            reference_data, comparison_data, common_columns, 
            categorical_columns, results, verbose
        )
        
        results = self._detect_distribution_drift(
            reference_data, comparison_data, numerical_columns, results, verbose
        )
        
        results = self._detect_statistical_drift(
            reference_data, comparison_data, numerical_columns, 
            categorical_columns, results, verbose
        )
        
        results["dimension_drift"] = self._detect_dimension_drift(
            reference_data, comparison_data, numerical_columns, verbose
        )
        
        results["outlier_drift"] = self._detect_outlier_drift(
            reference_data, comparison_data, numerical_columns, verbose
        )
        
        results["correlation_drift"] = self._detect_correlation_drift(
            reference_data, comparison_data, numerical_columns, verbose
        )
        
        # Calculate overall drift score
        results["drift_score"] = self._calculate_overall_drift_score(
            results, importance_weights
        )
        
        results["drift_detected"] = results["drift_score"] > self.drift_threshold
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        # Store results
        self.results_ = results
        
        # Display results if verbose
        if verbose:
            self._print_results(results)
        
        return results
    

    def _validate_and_prepare_data(self, 
                                  reference_data: pd.DataFrame,
                                  comparison_data: pd.DataFrame,
                                  date_columns: Optional[List[str]] = None) -> set:
        """Validate input data and prepare for analysis."""
        common_columns = set(reference_data.columns).intersection(
            set(comparison_data.columns)
        )
        
        if len(common_columns) == 0:
            raise ValueError("The two dataframes have no columns in common.")
        
        # Remove date columns if specified
        if date_columns:
            drop_cols = [col for col in date_columns if col in common_columns]
            common_columns = common_columns - set(drop_cols)
        
        return common_columns
    

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a series contains date-like data."""
        try:
            # Try to convert a sample to datetime
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            
            # Check if it looks like a date string
            sample_str = str(sample.iloc[0])
            if len(sample_str) >= 8 and ('-' in sample_str or '/' in sample_str):
                pd.to_datetime(sample, errors='raise')
                return True
            return False
        except:
            return False
    

    def _can_convert_to_numeric(self, series: pd.Series) -> bool:
        """Check if a series can be safely converted to numeric."""
        try:
            # Try converting a sample to numeric
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return False
            
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
    

    def _validate_numerical_columns(self, 
                                   reference_data: pd.DataFrame,
                                   comparison_data: pd.DataFrame,
                                   numerical_columns: List[str],
                                   verbose: bool) -> List[str]:
        """Validate that numerical columns can actually be converted to numeric."""
        valid_numerical = []
        
        for col in numerical_columns:
            ref_series = reference_data[col]
            comp_series = comparison_data[col]
            
            # Check if it's a date column
            if self._is_date_column(ref_series) or self._is_date_column(comp_series):
                if verbose:
                    print(f"Column '{col}' detected as date column, excluding from numerical analysis.")
                continue
            
            # Check if it can be converted to numeric
            if (self._can_convert_to_numeric(ref_series) and 
                self._can_convert_to_numeric(comp_series)):
                valid_numerical.append(col)
            else:
                if verbose:
                    print(f"Column '{col}' cannot be converted to numeric, excluding from numerical analysis.")
        
        return valid_numerical
    

    def _detect_categorical_columns(self, 
                                   reference_data: pd.DataFrame,
                                   comparison_data: pd.DataFrame,
                                   common_columns: set) -> List[str]:
        """Auto-detect categorical columns based on unique value count."""
        categorical_columns = []
        for col in common_columns:
            # Skip if it's a date column
            if (self._is_date_column(reference_data[col]) or 
                self._is_date_column(comparison_data[col])):
                continue
                
            # Check if it's categorical based on unique values
            if (reference_data[col].nunique() <= self.categorical_threshold and 
                comparison_data[col].nunique() <= self.categorical_threshold):
                categorical_columns.append(col)
            # Also check if it can't be converted to numeric
            elif (not self._can_convert_to_numeric(reference_data[col]) or
                  not self._can_convert_to_numeric(comparison_data[col])):
                categorical_columns.append(col)
                
        return categorical_columns
    

    def _initialize_results(self) -> Dict[str, Any]:
        """Initialize the results dictionary."""
        return {
            "drift_score": 0.0,
            "feature_drift": {},
            "distribution_drift": {},
            "statistical_drift": {},
            "dimension_drift": 0.0,
            "outlier_drift": 0.0,
            "correlation_drift": 0.0,
            "drift_detected": False,
            "recommendations": []
        }
    

    def _detect_feature_drift(self,
                             reference_data: pd.DataFrame,
                             comparison_data: pd.DataFrame,
                             common_columns: set,
                             categorical_columns: List[str],
                             results: Dict,
                             verbose: bool) -> Dict:
        """Detect drift at the feature level."""
        for col in common_columns:
            if reference_data[col].isna().all() or comparison_data[col].isna().all():
                continue
                
            if col in categorical_columns:
                drift_score = self._detect_categorical_drift(
                    reference_data[col], comparison_data[col], verbose
                )
            else:
                try:
                    drift_score = self._detect_numerical_drift(
                        reference_data[col], comparison_data[col], verbose
                    )
                except:
                    if verbose:
                        print(f"Column {col} failed numerical parsing, treating as categorical.")
                    categorical_columns.append(col)
                    drift_score = self._detect_categorical_drift(
                        reference_data[col], comparison_data[col], verbose
                    )
            
            results["feature_drift"][col] = drift_score
        
        return results
    

    def _detect_categorical_drift(self,
                                 series1: pd.Series,
                                 series2: pd.Series,
                                 verbose: bool = False) -> float:
        """Detect drift in categorical features using distribution comparison."""
        dist1 = series1.value_counts(normalize=True).to_dict()
        dist2 = series2.value_counts(normalize=True).to_dict()
        all_categories = set(dist1.keys()).union(set(dist2.keys()))
        
        # Jensen-Shannon distance
        js_distance = self._calculate_js_distance(dist1, dist2, all_categories)
        
        # Chi-square test
        chi2_drift_score = self._calculate_chi2_drift(
            dist1, dist2, all_categories, len(series1), len(series2)
        )
        
        # Population Stability Index
        psi = self._calculate_psi(dist1, dist2, all_categories)
        
        # Weighted average
        drift_score = (js_distance * 0.4 + chi2_drift_score * 0.3 + 
                      min(psi, 1) * 0.3)
        drift_score = np.clip(drift_score, 0.0, 1.0)
        
        if verbose:
            print(f"Categorical drift for {series1.name}: {drift_score:.4f}")
            print(f"  JS Distance: {js_distance:.4f}")
            print(f"  Chi-Square: {chi2_drift_score:.4f}")
            print(f"  PSI: {min(psi, 1):.4f}")
        
        return drift_score
    

    def _detect_numerical_drift(self,
                               series1: pd.Series,
                               series2: pd.Series,
                               verbose: bool = False) -> float:
        """Detect drift in numerical features using statistical tests."""
        # Convert to numeric, handling errors
        try:
            s1 = pd.to_numeric(series1, errors='coerce').dropna()
            s2 = pd.to_numeric(series2, errors='coerce').dropna()
        except:
            # If conversion fails, treat as categorical
            raise ValueError("Cannot convert to numeric")
        
        if len(s1) < 2 or len(s2) < 2:
            return 1.0
        
        # Kolmogorov-Smirnov test
        ks_drift = self._calculate_ks_drift(s1, s2)
        
        # Jensen-Shannon distance
        js_distance = self._calculate_js_distance_numerical(s1, s2)
        
        # Wasserstein distance
        emd_normalized = self._calculate_wasserstein_drift(s1, s2)
        
        # Basic statistics drift
        basic_stats_drift = self._calculate_basic_stats_drift(s1, s2)
        
        # Population Stability Index
        psi_normalized = self._calculate_psi_numerical(s1, s2)
        
        # Weighted combination
        drift_score = (ks_drift * 0.3 + js_distance * 0.25 + 
                      emd_normalized * 0.15 + basic_stats_drift * 0.1 + 
                      psi_normalized * 0.2)
        drift_score = np.clip(drift_score, 0.0, 1.0)
        
        if verbose:
            print(f"Numerical drift for {series1.name}: {drift_score:.4f}")
            print(f"  KS Test: {ks_drift:.4f}")
            print(f"  JS Distance: {js_distance:.4f}")
            print(f"  EMD: {emd_normalized:.4f}")
            print(f"  Stats Drift: {basic_stats_drift:.4f}")
            print(f"  PSI: {psi_normalized:.4f}")
        
        return drift_score
    

    def _detect_distribution_drift(self,
                                  reference_data: pd.DataFrame,
                                  comparison_data: pd.DataFrame,
                                  numerical_columns: List[str],
                                  results: Dict,
                                  verbose: bool) -> Dict:
        """Detect overall distribution drift between datasets."""
        distribution_results = {}
        
        # Joint distribution drift using KDE
        if len(numerical_columns) >= 2:
            try:
                sample_cols = numerical_columns[:2]
                # Convert to numeric and clean data
                ref_data = reference_data[sample_cols].apply(pd.to_numeric, errors='coerce').dropna()
                comp_data = comparison_data[sample_cols].apply(pd.to_numeric, errors='coerce').dropna()
                
                if len(ref_data) > 10 and len(comp_data) > 10:
                    kde1 = gaussian_kde(ref_data.values.T)
                    kde2 = gaussian_kde(comp_data.values.T)
                    
                    # Calculate Hellinger distance
                    h_dist = self._calculate_hellinger_distance(
                        kde1, kde2, ref_data, comp_data
                    )
                    distribution_results["joint_distribution"] = min(h_dist, 1.0)
                else:
                    distribution_results["joint_distribution"] = 0.5
            except:
                distribution_results["joint_distribution"] = 0.5
        else:
            distribution_results["joint_distribution"] = 0.0
        
        # Multivariate test
        if len(numerical_columns) >= 1:
            multivariate_drift = self._calculate_multivariate_drift(
                reference_data[numerical_columns], 
                comparison_data[numerical_columns]
            )
            distribution_results["multivariate_test"] = min(multivariate_drift, 1.0)
        else:
            distribution_results["multivariate_test"] = 0.0
        
        # Overall distribution drift
        distribution_results["overall"] = self._combine_distribution_scores(
            distribution_results
        )
        
        if verbose:
            print(f"Distribution drift: {distribution_results['overall']:.4f}")
            for k, v in distribution_results.items():
                if k != "overall":
                    print(f"  - {k}: {v:.4f}")
        
        results["distribution_drift"] = distribution_results
        return results
    

    def _detect_statistical_drift(self,
                                 reference_data: pd.DataFrame,
                                 comparison_data: pd.DataFrame,
                                 numerical_columns: List[str],
                                 categorical_columns: List[str],
                                 results: Dict,
                                 verbose: bool) -> Dict:
        """Run statistical tests to detect drift."""
        statistical_results = {}
        
        # Mann-Whitney U test for numerical columns
        if numerical_columns:
            u_scores = []
            for col in numerical_columns:
                try:
                    ref_data = pd.to_numeric(reference_data[col], errors='coerce').dropna()
                    comp_data = pd.to_numeric(comparison_data[col], errors='coerce').dropna()
                    
                    if len(ref_data) > 0 and len(comp_data) > 0:
                        _, p_value = stats.mannwhitneyu(
                            ref_data, comp_data, alternative='two-sided'
                        )
                        u_scores.append(1 - p_value)
                    else:
                        u_scores.append(0.5)
                except:
                    u_scores.append(0.5)
            
            if u_scores:
                statistical_results["mann_whitney_test"] = np.mean(u_scores)
        
        # Levene's test for variance equality
        if numerical_columns:
            levene_scores = []
            for col in numerical_columns:
                try:
                    ref_data = pd.to_numeric(reference_data[col], errors='coerce').dropna()
                    comp_data = pd.to_numeric(comparison_data[col], errors='coerce').dropna()
                    
                    if len(ref_data) > 0 and len(comp_data) > 0:
                        _, p_value = stats.levene(ref_data, comp_data)
                        levene_scores.append(1 - p_value)
                    else:
                        levene_scores.append(0.5)
                except:
                    levene_scores.append(0.5)
            
            if levene_scores:
                statistical_results["levene_test"] = np.mean(levene_scores)
        
        # Chi-squared test for categorical columns
        if categorical_columns:
            chi2_scores = []
            for col in categorical_columns:
                try:
                    table = pd.crosstab(
                        pd.Series(['ref'] * len(reference_data) + 
                                ['comp'] * len(comparison_data)),
                        pd.concat([reference_data[col], comparison_data[col]])
                    )
                    _, p_value, _, _ = stats.chi2_contingency(table)
                    chi2_scores.append(1 - p_value)
                except:
                    chi2_scores.append(0.5)
            
            if chi2_scores:
                statistical_results["chi2_test"] = np.mean(chi2_scores)
        
        # Calculate overall statistical drift
        statistical_results["overall"] = self._combine_statistical_scores(
            statistical_results
        )
        
        if verbose:
            print(f"Statistical drift: {statistical_results.get('overall', 0):.4f}")
            for k, v in statistical_results.items():
                if k != "overall":
                    print(f"  - {k}: {v:.4f}")
        
        results["statistical_drift"] = statistical_results
        return results
    

    def _detect_dimension_drift(self,
                               reference_data: pd.DataFrame,
                               comparison_data: pd.DataFrame,
                               numerical_columns: List[str],
                               verbose: bool) -> float:
        """Detect drift in dimensional structure using PCA."""
        if len(numerical_columns) < 2:
            return 0.0
        
        try:
            # Prepare data with proper numeric conversion
            X1 = reference_data[numerical_columns].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(method='ffill').fillna(0)
            X2 = comparison_data[numerical_columns].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(method='ffill').fillna(0)
            
            # Check if we have valid data
            if X1.empty or X2.empty or X1.isna().all().all() or X2.isna().all().all():
                if verbose:
                    print("Dimension drift: No valid numerical data available")
                return 0.0
            
            # Standardize data
            scaler = StandardScaler()
            X1_scaled = scaler.fit_transform(X1)
            X2_scaled = scaler.transform(X2)
            
            # Apply PCA
            n_components = min(3, len(numerical_columns))
            pca1 = PCA(n_components=n_components, random_state=self.random_state)
            pca2 = PCA(n_components=n_components, random_state=self.random_state)
            
            pca1.fit(X1_scaled)
            pca2.fit(X2_scaled)
            
            # Compare explained variance ratios
            variance_diff = np.mean(np.abs(
                pca1.explained_variance_ratio_ - pca2.explained_variance_ratio_
            ))
            
            # Compare principal components
            component_similarities = []
            for i in range(n_components):
                similarity = np.abs(cosine_similarity(
                    [pca1.components_[i]], [pca2.components_[i]]
                )[0][0])
                component_similarities.append(similarity)
            
            component_distances = [1 - sim for sim in component_similarities]
            weighted_distances = np.average(
                component_distances,
                weights=(pca1.explained_variance_ratio_ + 
                        pca2.explained_variance_ratio_) / 2
            )
            
            dimension_drift = 0.7 * weighted_distances + 0.3 * variance_diff
            dimension_drift = np.clip(dimension_drift, 0.0, 1.0)
            
            if verbose:
                print(f"Dimension drift: {dimension_drift:.4f}")
                print(f"  Component distance: {weighted_distances:.4f}")
                print(f"  Variance difference: {variance_diff:.4f}")
            
            return dimension_drift
        
        except Exception as e:
            if verbose:
                print(f"Dimension drift calculation failed: {e}")
            return 0.5
    

    def _detect_outlier_drift(self,
                             reference_data: pd.DataFrame,
                             comparison_data: pd.DataFrame,
                             numerical_columns: List[str],
                             verbose: bool) -> float:
        """Detect drift in outlier patterns using Isolation Forest."""
        if len(numerical_columns) < 1:
            return 0.0
        
        try:
            # Prepare data with proper numeric conversion
            X1 = reference_data[numerical_columns].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(method='ffill').fillna(0)
            X2 = comparison_data[numerical_columns].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(method='ffill').fillna(0)
            
            # Check if we have valid data
            if X1.empty or X2.empty or X1.isna().all().all() or X2.isna().all().all():
                if verbose:
                    print("Outlier drift: No valid numerical data available")
                return 0.0
            
            # Fit Isolation Forest on reference data
            iso_forest = IsolationForest(
                random_state=self.random_state, 
                contamination=0.1
            )
            iso_forest.fit(X1)
            
            # Calculate anomaly scores
            scores1 = -iso_forest.score_samples(X1)
            scores2 = -iso_forest.score_samples(X2)
            
            # Compare outlier rates
            thresh = np.percentile(scores1, 90)
            outlier_rate1 = np.mean(scores1 > thresh)
            outlier_rate2 = np.mean(scores2 > thresh)
            rate_diff = abs(outlier_rate2 - outlier_rate1)
            
            # Compare score distributions
            ks_stat, _ = stats.ks_2samp(scores1, scores2)
            
            # Compare mean scores
            mean_diff = abs(np.mean(scores1) - np.mean(scores2))
            mean_range = max(abs(np.mean(scores1)), abs(np.mean(scores2)))
            normalized_mean_diff = min(mean_diff / mean_range, 1.0) if mean_range > 0 else 0.0
            
            outlier_drift = (0.4 * rate_diff + 0.4 * ks_stat + 
                           0.2 * normalized_mean_diff)
            outlier_drift = np.clip(outlier_drift, 0.0, 1.0)
            
            if verbose:
                print(f"Outlier drift: {outlier_drift:.4f}")
                print(f"  Outlier rate diff: {rate_diff:.4f}")
                print(f"  KS statistic: {ks_stat:.4f}")
                print(f"  Mean score diff: {normalized_mean_diff:.4f}")
            
            return outlier_drift
        
        except Exception as e:
            if verbose:
                print(f"Outlier drift calculation failed: {e}")
            return 0.3
    

    def _detect_correlation_drift(self,
                                 reference_data: pd.DataFrame,
                                 comparison_data: pd.DataFrame,
                                 numerical_columns: List[str],
                                 verbose: bool) -> float:
        """Detect drift in correlation structure between features."""
        if len(numerical_columns) < 2:
            return 0.0
        
        try:
            # Prepare data with proper numeric conversion
            ref_data = reference_data[numerical_columns].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(method='ffill').fillna(0)
            comp_data = comparison_data[numerical_columns].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(method='ffill').fillna(0)
            
            # Check if we have valid data
            if ref_data.empty or comp_data.empty or ref_data.isna().all().all() or comp_data.isna().all().all():
                if verbose:
                    print("Correlation drift: No valid numerical data available")
                return 0.0
            
            # Calculate correlation matrices
            corr1 = ref_data.corr().fillna(0)
            corr2 = comp_data.corr().fillna(0)
            
            # Extract upper triangular elements
            corr1_vec = corr1.values[np.triu_indices(len(numerical_columns), k=1)]
            corr2_vec = corr2.values[np.triu_indices(len(numerical_columns), k=1)]
            
            # Frobenius norm of difference
            frob_norm = np.linalg.norm(corr1 - corr2) / np.sqrt(len(numerical_columns)**2)
            
            # Cosine similarity
            cos_sim = cosine_similarity([corr1_vec], [corr2_vec])[0][0]
            corr_distance = 1 - cos_sim
            
            # RV coefficient
            rv_distance = self._calculate_rv_coefficient(ref_data, comp_data)
            
            correlation_drift = (0.4 * frob_norm + 0.3 * corr_distance + 
                               0.3 * rv_distance)
            correlation_drift = np.clip(correlation_drift, 0.0, 1.0)
            
            if verbose:
                print(f"Correlation drift: {correlation_drift:.4f}")
                print(f"  Frobenius norm: {frob_norm:.4f}")
                print(f"  Correlation distance: {corr_distance:.4f}")
                print(f"  RV distance: {rv_distance:.4f}")
            
            return correlation_drift
        
        except Exception as e:
            if verbose:
                print(f"Correlation drift calculation failed: {e}")
            return 0.3
    

    def _calculate_overall_drift_score(self,
                                     results: Dict,
                                     importance_weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate the overall drift score."""
        feature_drift_scores = results["feature_drift"]
        
        if importance_weights:
            total_weight = sum(importance_weights.values())
            normalized_weights = {k: v/total_weight for k, v in importance_weights.items()}
            weighted_feature_drift = 0
            for col, score in feature_drift_scores.items():
                weight = normalized_weights.get(col, 1 / len(feature_drift_scores))
                weighted_feature_drift += score * weight
        else:
            weighted_feature_drift = (np.mean(list(feature_drift_scores.values())) 
                                    if feature_drift_scores else 0)
        
        # Ensemble drift score
        overall_drift = (
            weighted_feature_drift * 0.4 +
            results["distribution_drift"].get("overall", 0) * 0.2 +
            results["statistical_drift"].get("overall", 0) * 0.15 +
            results["dimension_drift"] * 0.1 +
            results["outlier_drift"] * 0.05 +
            results["correlation_drift"] * 0.1
        )
        
        return np.clip(overall_drift, 0.0, 1.0)
    

    def _generate_recommendations(self, results: Dict) -> List[str]:
            """Generate actionable recommendations based on drift analysis."""
            recommendations = []
            
            # High drift score recommendations
            if results["drift_score"] > 0.8:
                recommendations.append("CRITICAL: High drift detected. Consider retraining your model immediately.")
            elif results["drift_score"] > self.drift_threshold:
                recommendations.append("WARNING: Significant drift detected. Monitor model performance closely.")
            
            # Feature-specific recommendations
            high_drift_features = [col for col, score in results["feature_drift"].items() 
                                if score > 0.7]
            if high_drift_features:
                recommendations.append(f"High drift detected in features: {', '.join(high_drift_features)}. "
                                    "Consider feature engineering or data preprocessing adjustments.")
            
            # Distribution drift recommendations
            if results["distribution_drift"].get("overall", 0) > 0.7:
                recommendations.append("Significant distribution drift detected. "
                                    "Check data collection process and preprocessing pipeline.")
            
            # Outlier drift recommendations
            if results["outlier_drift"] > 0.6:
                recommendations.append("Outlier patterns have changed. "
                                    "Review data quality and consider outlier detection mechanisms.")
            
            # Correlation drift recommendations
            if results["correlation_drift"] > 0.6:
                recommendations.append("Feature relationships have changed. "
                                    "Consider updating feature selection or model architecture.")
            
            # General recommendations
            if results["drift_detected"]:
                recommendations.extend([
                    "Increase monitoring frequency for this data source.",
                    "Consider implementing automated retraining triggers.",
                    "Evaluate model performance metrics on recent data."
                ])
            
            return recommendations


    def _print_results(self, results: Dict):
        """Print comprehensive drift analysis results."""
        print("\n" + "="*60)
        print("DATA DRIFT ANALYSIS RESULTS")
        print("="*60)
        
        # Overall results
        print(f"\nOverall Drift Score: {results['drift_score']:.4f}")
        print(f"Drift Detected: {'YES' if results['drift_detected'] else 'NO'}")
        print(f"Threshold: {self.drift_threshold}")
        
        # Feature-level drift
        print(f"\n{'Feature Drift Analysis':<30} {'Score':<10}")
        print("-" * 40)
        for feature, score in results["feature_drift"].items():
            status = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
            print(f"{feature:<30} {score:<10.4f} [{status}]")
        
        # Other drift types
        print(f"\n{'Drift Type':<25} {'Score':<10} {'Status':<10}")
        print("-" * 45)
        
        dist_score = results["distribution_drift"].get("overall", 0)
        stat_score = results["statistical_drift"].get("overall", 0)
        
        for name, score in [
            ("Distribution", dist_score),
            ("Statistical", stat_score),
            ("Dimensional", results["dimension_drift"]),
            ("Outlier", results["outlier_drift"]),
            ("Correlation", results["correlation_drift"])
        ]:
            status = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
            print(f"{name:<25} {score:<10.4f} [{status}]")
        
        # Recommendations
        if results["recommendations"]:
            print(f"\nRECOMMENDATIONS:")
            print("-" * 20)
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*60)
    

    def _calculate_js_distance(self, dist1: Dict, dist2: Dict, all_categories: set) -> float:
        """Calculate Jensen-Shannon distance between two categorical distributions."""
        # Create probability vectors
        p = np.array([dist1.get(cat, 0) for cat in all_categories])
        q = np.array([dist2.get(cat, 0) for cat in all_categories])
        
        # Normalize
        p = p / np.sum(p) if np.sum(p) > 0 else np.ones(len(p)) / len(p)
        q = q / np.sum(q) if np.sum(q) > 0 else np.ones(len(q)) / len(q)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate JS distance
        m = 0.5 * (p + q)
        js_distance = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return min(np.sqrt(js_distance), 1.0)
    

    def _calculate_chi2_drift(self, dist1: Dict, dist2: Dict, all_categories: set, 
                            n1: int, n2: int) -> float:
        """Calculate chi-square based drift score."""
        try:
            observed1 = np.array([dist1.get(cat, 0) * n1 for cat in all_categories])
            observed2 = np.array([dist2.get(cat, 0) * n2 for cat in all_categories])
            
            # Create contingency table
            contingency = np.array([observed1, observed2])
            
            # Avoid zero counts
            contingency = contingency + 0.5
            
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            return min(1 - p_value, 1.0)
        except:
            return 0.5
    

    def _calculate_psi(self, dist1: Dict, dist2: Dict, all_categories: set) -> float:
        """Calculate Population Stability Index."""
        psi = 0
        for cat in all_categories:
            p1 = dist1.get(cat, 1e-10)
            p2 = dist2.get(cat, 1e-10)
            
            # Avoid division by zero
            if p1 == 0:
                p1 = 1e-10
            if p2 == 0:
                p2 = 1e-10
                
            psi += (p2 - p1) * np.log(p2 / p1)
        
        return abs(psi)
    

    def _calculate_ks_drift(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate Kolmogorov-Smirnov test based drift."""
        try:
            ks_stat, _ = stats.ks_2samp(series1, series2)
            return min(ks_stat, 1.0)
        except:
            return 0.5
    

    def _calculate_js_distance_numerical(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate JS distance for numerical data using histograms."""
        try:
            # Create histograms with same bins
            min_val = min(series1.min(), series2.min())
            max_val = max(series1.max(), series2.max())
            bins = np.linspace(min_val, max_val, 20)
            
            hist1, _ = np.histogram(series1, bins=bins, density=True)
            hist2, _ = np.histogram(series2, bins=bins, density=True)
            
            # Normalize
            hist1 = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else np.ones(len(hist1)) / len(hist1)
            hist2 = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else np.ones(len(hist2)) / len(hist2)
            
            # Add epsilon
            epsilon = 1e-10
            hist1 = hist1 + epsilon
            hist2 = hist2 + epsilon
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # JS distance
            m = 0.5 * (hist1 + hist2)
            js_distance = 0.5 * np.sum(hist1 * np.log(hist1 / m)) + 0.5 * np.sum(hist2 * np.log(hist2 / m))
            return min(np.sqrt(js_distance), 1.0)
        except:
            return 0.5
    

    def _calculate_wasserstein_drift(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate normalized Wasserstein (Earth Mover's) distance."""
        try:
            emd = stats.wasserstein_distance(series1, series2)
            # Normalize by the range of values
            value_range = max(series1.max(), series2.max()) - min(series1.min(), series2.min())
            if value_range > 0:
                return min(emd / value_range, 1.0)
            else:
                return 0.0
        except:
            return 0.5
    

    def _calculate_basic_stats_drift(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate drift based on basic statistical measures."""
        try:
            # Calculate relative differences in basic statistics
            stats1 = [series1.mean(), series1.std(), series1.skew(), series1.kurt()]
            stats2 = [series2.mean(), series2.std(), series2.skew(), series2.kurt()]
            
            relative_diffs = []
            for s1, s2 in zip(stats1, stats2):
                if abs(s1) > 1e-10 or abs(s2) > 1e-10:
                    max_val = max(abs(s1), abs(s2))
                    relative_diffs.append(abs(s1 - s2) / max_val)
                else:
                    relative_diffs.append(0.0)
            
            return min(np.mean(relative_diffs), 1.0)
        except:
            return 0.5
    

    def _calculate_psi_numerical(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate PSI for numerical data using binning."""
        try:
            # Create bins based on reference data quantiles
            bins = np.percentile(series1, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            bins = np.unique(bins)  # Remove duplicates
            
            if len(bins) < 3:
                return 0.5
            
            # Calculate distributions
            hist1, _ = np.histogram(series1, bins=bins, density=False)
            hist2, _ = np.histogram(series2, bins=bins, density=False)
            
            # Convert to proportions
            p1 = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else np.ones(len(hist1)) / len(hist1)
            p2 = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else np.ones(len(hist2)) / len(hist2)
            
            # Calculate PSI
            psi = 0
            for i in range(len(p1)):
                if p1[i] == 0:
                    p1[i] = 1e-10
                if p2[i] == 0:
                    p2[i] = 1e-10
                psi += (p2[i] - p1[i]) * np.log(p2[i] / p1[i])
            
            return min(abs(psi), 1.0)
        except:
            return 0.5
    

    def _calculate_hellinger_distance(self, kde1, kde2, ref_data: pd.DataFrame, 
                                    comp_data: pd.DataFrame) -> float:
        """Calculate Hellinger distance between two KDE distributions."""
        try:
            # Create grid for evaluation
            x_min = min(ref_data.iloc[:, 0].min(), comp_data.iloc[:, 0].min())
            x_max = max(ref_data.iloc[:, 0].max(), comp_data.iloc[:, 0].max())
            y_min = min(ref_data.iloc[:, 1].min(), comp_data.iloc[:, 1].min())
            y_max = max(ref_data.iloc[:, 1].max(), comp_data.iloc[:, 1].max())
            
            x_grid = np.linspace(x_min, x_max, 20)
            y_grid = np.linspace(y_min, y_max, 20)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([X.ravel(), Y.ravel()])
            
            # Evaluate KDEs
            f1 = kde1(positions).reshape(X.shape)
            f2 = kde2(positions).reshape(X.shape)
            
            # Normalize
            f1 = f1 / np.sum(f1)
            f2 = f2 / np.sum(f2)
            f1 = f1 + 1e-10
            f2 = f2 + 1e-10
            f1 = f1 / np.sum(f1)
            f2 = f2 / np.sum(f2)
            
            # Hellinger distance
            hellinger = np.sqrt(0.5 * np.sum((np.sqrt(f1) - np.sqrt(f2))**2))
            return min(hellinger, 1.0)
        except:
            return 0.5
    

    def _calculate_multivariate_drift(self, ref_data: pd.DataFrame, 
                                    comp_data: pd.DataFrame) -> float:
        """Calculate multivariate drift using statistical tests."""
        try:
            # Convert to numeric
            ref_numeric = ref_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            comp_numeric = comp_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            if ref_numeric.empty or comp_numeric.empty:
                return 0.5
            
            # Sample data if too large
            if len(ref_numeric) > 1000:
                ref_numeric = ref_numeric.sample(1000, random_state=self.random_state)
            if len(comp_numeric) > 1000:
                comp_numeric = comp_numeric.sample(1000, random_state=self.random_state)
            
            # Combine datasets with labels
            combined = np.vstack([ref_numeric.values, comp_numeric.values])
            labels = np.hstack([np.zeros(len(ref_numeric)), np.ones(len(comp_numeric))])
            
            # Simple multivariate test using mean differences
            ref_mean = np.mean(ref_numeric.values, axis=0)
            comp_mean = np.mean(comp_numeric.values, axis=0)
            
            # Calculate normalized distance
            diff = ref_mean - comp_mean
            ref_std = np.std(ref_numeric.values, axis=0) + 1e-10
            normalized_diff = np.abs(diff / ref_std)
            
            return min(np.mean(normalized_diff), 1.0)
        except:
            return 0.5
    

    def _calculate_rv_coefficient(self, ref_data: pd.DataFrame, 
                                comp_data: pd.DataFrame) -> float:
        """Calculate RV coefficient for correlation structure comparison."""
        try:
            # Calculate cross-product matrices
            ref_centered = ref_data - ref_data.mean()
            comp_centered = comp_data - comp_data.mean()
            
            S1 = np.dot(ref_centered.T, ref_centered)
            S2 = np.dot(comp_centered.T, comp_centered)
            
            # RV coefficient
            numerator = np.trace(np.dot(S1, S2))
            denominator = np.sqrt(np.trace(np.dot(S1, S1)) * np.trace(np.dot(S2, S2)))
            
            if denominator > 0:
                rv_coeff = numerator / denominator
                return 1 - abs(rv_coeff)
            else:
                return 0.5
        except:
            return 0.5
    

    def _combine_distribution_scores(self, distribution_results: Dict) -> float:
        """Combine distribution drift scores."""
        scores = [v for k, v in distribution_results.items() if k != "overall"]
        return np.mean(scores) if scores else 0.0
    

    def _combine_statistical_scores(self, statistical_results: Dict) -> float:
        """Combine statistical drift scores."""
        scores = [v for k, v in statistical_results.items() if k != "overall"]
        return np.mean(scores) if scores else 0.0
    

    def get_drift_summary(self) -> Dict:
        """Get a summary of the drift analysis results."""
        if self.results_ is None:
            raise ValueError("No drift analysis has been performed. Call detect_drift() first.")
        
        return {
            "overall_drift_score": self.results_["drift_score"],
            "drift_detected": self.results_["drift_detected"],
            "high_drift_features": [col for col, score in self.results_["feature_drift"].items() 
                                if score > 0.7],
            "moderate_drift_features": [col for col, score in self.results_["feature_drift"].items() 
                                    if 0.4 < score <= 0.7],
            "num_recommendations": len(self.results_["recommendations"])
        }
    