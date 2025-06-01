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
        
        numerical_columns = [col for col in common_columns 
                           if col not in categorical_columns]
        
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
    
    def _detect_categorical_columns(self, 
                                   reference_data: pd.DataFrame,
                                   comparison_data: pd.DataFrame,
                                   common_columns: set) -> List[str]:
        """Auto-detect categorical columns based on unique value count."""
        categorical_columns = []
        for col in common_columns:
            if (reference_data[col].nunique() <= self.categorical_threshold and 
                comparison_data[col].nunique() <= self.categorical_threshold):
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
        s1 = series1.dropna()
        s2 = series2.dropna()
        
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
                kde1 = gaussian_kde(reference_data[sample_cols].dropna().values.T)
                kde2 = gaussian_kde(comparison_data[sample_cols].dropna().values.T)
                
                # Calculate Hellinger distance
                h_dist = self._calculate_hellinger_distance(
                    kde1, kde2, reference_data[sample_cols], 
                    comparison_data[sample_cols]
                )
                distribution_results["joint_distribution"] = min(h_dist, 1.0)
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
                    _, p_value = stats.mannwhitneyu(
                        reference_data[col].dropna(),
                        comparison_data[col].dropna(),
                        alternative='two-sided'
                    )
                    u_scores.append(1 - p_value)
                except:
                    u_scores.append(0.5)
            
            if u_scores:
                statistical_results["mann_whitney_test"] = np.mean(u_scores)
        
        # Levene's test for variance equality
        if numerical_columns:
            levene_scores = []
            for col in numerical_columns:
                try:
                    _, p_value = stats.levene(
                        reference_data[col].dropna(),
                        comparison_data[col].dropna()
                    )
                    levene_scores.append(1 - p_value)
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
            # Prepare data
            X1 = reference_data[numerical_columns].fillna(
                reference_data[numerical_columns].mean()
            )
            X2 = comparison_data[numerical_columns].fillna(
                comparison_data[numerical_columns].mean()
            )
            
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
            # Prepare data
            X1 = reference_data[numerical_columns].fillna(
                reference_data[numerical_columns].mean()
            )
            X2 = comparison_data[numerical_columns].fillna(
                comparison_data[numerical_columns].mean()
            )
            
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
            # Calculate correlation matrices
            corr1 = reference_data[numerical_columns].corr().fillna(0)
            corr2 = comparison_data[numerical_columns].corr().fillna(0)
            
            # Extract upper triangular elements
            corr1_vec = corr1.values[np.triu_indices(len(numerical_columns), k=1)]
            corr2_vec = corr2.values[np.triu_indices(len(numerical_columns), k=1)]
            
            # Frobenius norm of difference
            frob_norm = np.linalg.norm(corr1 - corr2) / np.sqrt(len(numerical_columns)**2)
            
            # Cosine similarity
            cos_sim = cosine_similarity([corr1_vec], [corr2_vec])[0][0]
            corr_distance = 1 - cos_sim
            
            # RV coefficient
            rv_distance = self._calculate_rv_coefficient(
                reference_data[numerical_columns],
                comparison_data[numerical_columns]
            )
            
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
        feature_drift_scores = results["feature_drift"]
        sorted_features = sorted(feature_drift_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        if results["drift_score"] > self.drift_threshold:
            recommendations.append(
                "Significant drift detected. Consider retraining your model with newer data."
            )
            
            top_drifting = [f[0] for f in sorted_features[:3] 
                          if f[1] > self.drift_threshold]
            if top_drifting:
                recommendations.append(
                    f"Focus on these high-drift features: {', '.join(top_drifting)}."
                )
            
            if results["correlation_drift"] > self.drift_threshold:
                recommendations.append(
                    "Significant changes in feature correlations detected. "
                    "Review feature interactions."
                )
            
            if results["dimension_drift"] > self.drift_threshold:
                recommendations.append(
                    "Underlying data dimensions have shifted. "
                    "Consider revisiting feature engineering."
                )
            
            if results["outlier_drift"] > self.drift_threshold:
                recommendations.append(
                    "Outlier patterns have changed. "
                    "Review outlier handling strategy."
                )
        
        elif results["drift_score"] > self.drift_threshold * 0.7:
            recommendations.append(
                "Moderate drift detected. Monitor model performance closely."
            )
            
            mod_drifting = [f[0] for f in sorted_features[:2] 
                          if f[1] > self.drift_threshold * 0.7]
            if mod_drifting:
                recommendations.append(
                    f"Keep an eye on these features: {', '.join(mod_drifting)}."
                )
        
        else:
            recommendations.append(
                "No significant drift detected. "
                "Model should perform similarly on new data."
            )
        
        return recommendations
    
    def _print_results(self, results: Dict) -> None:
        """Print detailed drift analysis results."""
        print(f"Overall drift score: {results['drift_score']:.4f}")
        print(f"Drift detected: {results['drift_detected']}")
        print("\nTop drifting features:")
        
        top_features = sorted(results["feature_drift"].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        for feature, score in top_features:
            print(f"  - {feature}: {score:.4f}")
        
        print("\nRecommendations:")
        for rec in results["recommendations"]:
            print(f"  - {rec}")
    
    # Helper methods for calculations (keeping them concise)
    def _calculate_js_distance(self, dist1: Dict, dist2: Dict, categories: set) -> float:
        """Calculate Jensen-Shannon distance between two distributions."""
        epsilon = 1e-10
        js_distance = 0
        
        for category in categories:
            p = dist1.get(category, 0) + epsilon
            q = dist2.get(category, 0) + epsilon
            m = (p + q) / 2
            js_distance += 0.5 * (p * np.log(p/m) + q * np.log(q/m))
        
        return js_distance
    
    def _calculate_chi2_drift(self, dist1: Dict, dist2: Dict, categories: set, 
                             n1: int, n2: int) -> float:
        """Calculate chi-square drift score."""
        try:
            counts1 = [dist1.get(cat, 0) * n1 for cat in categories]
            counts2 = [dist2.get(cat, 0) * n2 for cat in categories]
            
            if all(c >= 5 for c in counts1 + counts2):
                _, p_value = stats.chi2_contingency([counts1, counts2])
                return 1 - p_value
            else:
                return self._calculate_js_distance(dist1, dist2, categories)
        except:
            return self._calculate_js_distance(dist1, dist2, categories)
    
    def _calculate_psi(self, dist1: Dict, dist2: Dict, categories: set) -> float:
        """Calculate Population Stability Index."""
        epsilon = 1e-10
        psi = 0
        
        for category in categories:
            p = dist1.get(category, 0) + epsilon
            q = dist2.get(category, 0) + epsilon
            psi += (p - q) * np.log(p / q)
        
        return psi
    
    def _calculate_ks_drift(self, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate Kolmogorov-Smirnov drift score."""
        try:
            _, p_value = stats.ks_2samp(s1, s2)
            return 1 - p_value
        except:
            return 0.5

    def _calculate_js_distance_numerical(self, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate Jensen-Shannon distance for numerical data."""
        try:
            hist1, bin_edges = np.histogram(s1, bins=20, density=True)
            hist2, _ = np.histogram(s2, bins=bin_edges, density=True)
            
            epsilon = 1e-10
            hist1 = hist1 + epsilon
            hist2 = hist2 + epsilon
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            m = (hist1 + hist2) / 2
            js_distance = 0.5 * (np.sum(hist1 * np.log(hist1 / m)) + 
                                np.sum(hist2 * np.log(hist2 / m)))
            
            return np.clip(js_distance, 0.0, 1.0)
        except:
            return 0.5
    
    def _calculate_wasserstein_drift(self, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate Wasserstein (Earth Mover's) distance."""
        try:
            emd = stats.wasserstein_distance(s1, s2)
            # Normalize by data range
            data_range = max(s1.max(), s2.max()) - min(s1.min(), s2.min())
            if data_range > 0:
                emd_normalized = min(emd / data_range, 1.0)
            else:
                emd_normalized = 0.0
            return emd_normalized
        except:
            return 0.5
    
    def _calculate_basic_stats_drift(self, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate drift based on basic statistical differences."""
        try:
            stats1 = {
                'mean': s1.mean(),
                'std': s1.std(),
                'skew': stats.skew(s1),
                'kurtosis': stats.kurtosis(s1)
            }
            
            stats2 = {
                'mean': s2.mean(),
                'std': s2.std(),
                'skew': stats.skew(s2),
                'kurtosis': stats.kurtosis(s2)
            }
            
            differences = []
            for stat in ['mean', 'std', 'skew', 'kurtosis']:
                val1, val2 = stats1[stat], stats2[stat]
                if np.isnan(val1) or np.isnan(val2):
                    continue
                
                if stat in ['mean', 'std']:
                    # Normalize by data range for mean and std
                    data_range = max(s1.max(), s2.max()) - min(s1.min(), s2.min())
                    if data_range > 0:
                        diff = abs(val1 - val2) / data_range
                    else:
                        diff = 0.0
                else:
                    # For skew and kurtosis, use relative difference
                    if abs(val1) + abs(val2) > 0:
                        diff = abs(val1 - val2) / (abs(val1) + abs(val2))
                    else:
                        diff = 0.0
                
                differences.append(min(diff, 1.0))
            
            return np.mean(differences) if differences else 0.0
        except:
            return 0.5
    
    def _calculate_psi_numerical(self, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate Population Stability Index for numerical data."""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(s1, bins=10)
            
            # Calculate bin counts
            counts1, _ = np.histogram(s1, bins=bin_edges)
            counts2, _ = np.histogram(s2, bins=bin_edges)
            
            # Convert to proportions
            prop1 = counts1 / len(s1)
            prop2 = counts2 / len(s2)
            
            # Calculate PSI
            epsilon = 1e-10
            psi = 0
            for i in range(len(prop1)):
                p = prop1[i] + epsilon
                q = prop2[i] + epsilon
                psi += (p - q) * np.log(p / q)
            
            return min(abs(psi), 1.0)
        except:
            return 0.5
    
    def _calculate_hellinger_distance(self, kde1, kde2, data1: pd.DataFrame, 
                                    data2: pd.DataFrame) -> float:
        """Calculate Hellinger distance between two KDE distributions."""
        try:
            # Create a grid for evaluation
            x_min = min(data1.iloc[:, 0].min(), data2.iloc[:, 0].min())
            x_max = max(data1.iloc[:, 0].max(), data2.iloc[:, 0].max())
            y_min = min(data1.iloc[:, 1].min(), data2.iloc[:, 1].min())
            y_max = max(data1.iloc[:, 1].max(), data2.iloc[:, 1].max())
            
            x = np.linspace(x_min, x_max, 20)
            y = np.linspace(y_min, y_max, 20)
            xx, yy = np.meshgrid(x, y)
            grid = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate KDEs
            pdf1 = kde1(grid)
            pdf2 = kde2(grid)
            
            # Normalize
            pdf1 = pdf1 / np.sum(pdf1)
            pdf2 = pdf2 / np.sum(pdf2)
            
            # Calculate Hellinger distance
            hellinger = np.sqrt(0.5 * np.sum((np.sqrt(pdf1) - np.sqrt(pdf2))**2))
            return hellinger
        except:
            return 0.5
    
    def _calculate_multivariate_drift(self, X1: pd.DataFrame, X2: pd.DataFrame) -> float:
        """Calculate multivariate drift using energy statistics."""
        try:
            # Sample data if too large
            n1, n2 = len(X1), len(X2)
            if n1 > 1000:
                X1 = X1.sample(n=1000, random_state=self.random_state)
            if n2 > 1000:
                X2 = X2.sample(n=1000, random_state=self.random_state)
            
            # Fill NaN values
            X1_clean = X1.fillna(X1.mean())
            X2_clean = X2.fillna(X2.mean())
            
            # Standardize
            scaler = StandardScaler()
            X1_scaled = scaler.fit_transform(X1_clean)
            X2_scaled = scaler.transform(X2_clean)
            
            # Calculate energy statistic approximation
            n1, n2 = len(X1_scaled), len(X2_scaled)
            
            # Calculate pairwise distances within and between samples
            def avg_distance(X, Y):
                if len(X) == 0 or len(Y) == 0:
                    return 0
                distances = []
                for i in range(min(len(X), 100)):  # Sample for efficiency
                    for j in range(min(len(Y), 100)):
                        distances.append(np.linalg.norm(X[i] - Y[j]))
                return np.mean(distances)
            
            d_12 = avg_distance(X1_scaled, X2_scaled)
            d_11 = avg_distance(X1_scaled, X1_scaled)
            d_22 = avg_distance(X2_scaled, X2_scaled)
            
            # Energy statistic
            energy_stat = 2 * d_12 - d_11 - d_22
            
            # Normalize to [0, 1]
            energy_normalized = min(abs(energy_stat), 1.0)
            
            return energy_normalized
        except:
            return 0.5
    
    def _combine_distribution_scores(self, distribution_results: Dict) -> float:
        """Combine multiple distribution drift scores."""
        scores = [v for k, v in distribution_results.items() if k != "overall"]
        if not scores:
            return 0.0
        return np.mean(scores)
    
    def _combine_statistical_scores(self, statistical_results: Dict) -> float:
        """Combine multiple statistical test scores."""
        scores = [v for k, v in statistical_results.items() if k != "overall"]
        if not scores:
            return 0.0
        return np.mean(scores)
    
    def _calculate_rv_coefficient(self, X1: pd.DataFrame, X2: pd.DataFrame) -> float:
        """Calculate RV coefficient distance for correlation comparison."""
        try:
            # Fill NaN values
            X1_clean = X1.fillna(X1.mean())
            X2_clean = X2.fillna(X2.mean())
            
            # Calculate correlation matrices
            R1 = X1_clean.corr().fillna(0).values
            R2 = X2_clean.corr().fillna(0).values
            
            # Calculate RV coefficient
            numerator = np.trace(R1 @ R2)
            denominator = np.sqrt(np.trace(R1 @ R1) * np.trace(R2 @ R2))
            
            if denominator > 0:
                rv_coeff = numerator / denominator
                rv_distance = 1 - abs(rv_coeff)
            else:
                rv_distance = 0.5
            
            return np.clip(rv_distance, 0.0, 1.0)
        except:
            return 0.5
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the drift detection results.
        
        Returns:
        --------
        Dict[str, Any]
            Summary dictionary with key metrics and insights
        """
        if self.results_ is None:
            return {"error": "No drift detection results available. Run detect_drift() first."}
        
        feature_drift = self.results_["feature_drift"]
        
        # Calculate summary statistics
        high_drift_features = [f for f, score in feature_drift.items() 
                              if score > self.drift_threshold]
        moderate_drift_features = [f for f, score in feature_drift.items() 
                                  if self.drift_threshold * 0.5 < score <= self.drift_threshold]
        
        top_drifting = sorted(feature_drift.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = {
            "overall_drift_score": round(self.results_["drift_score"], 4),
            "drift_detected": self.results_["drift_detected"],
            "total_features_analyzed": len(feature_drift),
            "high_drift_features_count": len(high_drift_features),
            "moderate_drift_features_count": len(moderate_drift_features),
            "stable_features_count": len(feature_drift) - len(high_drift_features) - len(moderate_drift_features),
            "top_drifting_features": {f: round(score, 4) for f, score in top_drifting},
            "drift_components": {
                "dimension_drift": round(self.results_["dimension_drift"], 4),
                "outlier_drift": round(self.results_["outlier_drift"], 4),
                "correlation_drift": round(self.results_["correlation_drift"], 4),
                "distribution_drift": round(self.results_["distribution_drift"].get("overall", 0), 4),
                "statistical_drift": round(self.results_["statistical_drift"].get("overall", 0), 4)
            },
            "recommendations": self.results_["recommendations"],
            "drift_threshold_used": self.drift_threshold
        }
        
        return summary
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export drift detection results to file.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the results
        format : str, default='json'
            Export format ('json', 'csv', or 'pickle')
        """
        if self.results_ is None:
            raise ValueError("No drift detection results available. Run detect_drift() first.")
        
        if format.lower() == 'json':
            import json
            # Convert numpy types to Python types for JSON serialization
            json_results = self._convert_for_json(self.results_)
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
        
        elif format.lower() == 'csv':
            # Export feature drift scores as CSV
            feature_drift_df = pd.DataFrame(
                list(self.results_["feature_drift"].items()),
                columns=['Feature', 'Drift_Score']
            )
            feature_drift_df.to_csv(filepath, index=False)
        
        elif format.lower() == 'pickle':
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.results_, f)
        
        else:
            raise ValueError("Supported formats are: 'json', 'csv', 'pickle'")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj