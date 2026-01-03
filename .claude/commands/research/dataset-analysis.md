# ML Dataset Analysis

Comprehensive exploratory data analysis for machine learning datasets with focus on data quality, distribution analysis, and ML-readiness assessment.

## Requirements

- **Dataset**: $ARGUMENTS (provide dataset path, format, or description of the data)

## Instructions

### 1. Initial Data Assessment

**A. Load and Inspect Structure:**
```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

class DatasetAnalyzer:
    """Comprehensive ML dataset analyzer."""

    def __init__(self, data: pd.DataFrame, target_col: Optional[str] = None):
        self.data = data
        self.target_col = target_col
        self.analysis_results = {}

    def basic_info(self) -> Dict[str, Any]:
        """Get basic dataset information."""
        info = {
            "n_samples": len(self.data),
            "n_features": len(self.data.columns),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "dtypes": self.data.dtypes.value_counts().to_dict(),
            "columns": {
                col: {
                    "dtype": str(self.data[col].dtype),
                    "n_unique": self.data[col].nunique(),
                    "n_missing": self.data[col].isna().sum(),
                    "missing_pct": self.data[col].isna().mean() * 100
                }
                for col in self.data.columns
            }
        }
        self.analysis_results["basic_info"] = info
        return info

    def print_summary(self):
        """Print formatted summary."""
        info = self.basic_info()
        print(f"Dataset Shape: {info['n_samples']:,} samples × {info['n_features']} features")
        print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        print(f"\nData Types:")
        for dtype, count in info['dtypes'].items():
            print(f"  {dtype}: {count}")

        missing = [(col, d['missing_pct']) for col, d in info['columns'].items()
                   if d['missing_pct'] > 0]
        if missing:
            print(f"\nMissing Values ({len(missing)} columns):")
            for col, pct in sorted(missing, key=lambda x: -x[1])[:10]:
                print(f"  {col}: {pct:.1f}%")
```

**B. Numerical Feature Analysis:**
```python
def analyze_numerical(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Analyze numerical features."""
    if columns is None:
        columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

    stats = []
    for col in columns:
        series = self.data[col].dropna()
        stat = {
            "column": col,
            "count": len(series),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "25%": series.quantile(0.25),
            "50%": series.quantile(0.50),
            "75%": series.quantile(0.75),
            "max": series.max(),
            "skewness": series.skew(),
            "kurtosis": series.kurtosis(),
            "n_zeros": (series == 0).sum(),
            "n_negative": (series < 0).sum(),
        }

        # Detect outliers (IQR method)
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
        stat["n_outliers_iqr"] = outliers
        stat["outlier_pct"] = outliers / len(series) * 100

        stats.append(stat)

    return pd.DataFrame(stats).set_index("column")
```

**C. Categorical Feature Analysis:**
```python
def analyze_categorical(self, columns: Optional[List[str]] = None,
                        max_categories: int = 50) -> Dict[str, Any]:
    """Analyze categorical features."""
    if columns is None:
        columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

    results = {}
    for col in columns:
        series = self.data[col]
        value_counts = series.value_counts()

        results[col] = {
            "n_unique": series.nunique(),
            "n_missing": series.isna().sum(),
            "most_common": value_counts.head(5).to_dict(),
            "least_common": value_counts.tail(5).to_dict(),
            "entropy": self._entropy(series.dropna()),
            "is_high_cardinality": series.nunique() > max_categories,
        }

        # Check for potential issues
        results[col]["issues"] = []
        if series.nunique() == len(series):
            results[col]["issues"].append("All unique (potential ID column)")
        if series.nunique() == 1:
            results[col]["issues"].append("Single value (no variance)")
        if series.nunique() > max_categories:
            results[col]["issues"].append(f"High cardinality ({series.nunique()} categories)")

    return results

def _entropy(self, series: pd.Series) -> float:
    """Calculate entropy of categorical distribution."""
    probs = series.value_counts(normalize=True)
    return -(probs * np.log2(probs + 1e-10)).sum()
```

### 2. Target Variable Analysis

```python
def analyze_target(self) -> Dict[str, Any]:
    """Analyze target variable for ML readiness."""
    if self.target_col is None:
        return {"error": "No target column specified"}

    target = self.data[self.target_col]

    analysis = {
        "dtype": str(target.dtype),
        "n_unique": target.nunique(),
        "n_missing": target.isna().sum(),
    }

    # Classification vs Regression detection
    if target.dtype in ['object', 'category'] or target.nunique() < 20:
        analysis["task_type"] = "classification"
        analysis["class_distribution"] = target.value_counts().to_dict()
        analysis["class_balance"] = target.value_counts(normalize=True).to_dict()

        # Imbalance metrics
        counts = target.value_counts()
        analysis["imbalance_ratio"] = counts.max() / counts.min()
        analysis["is_imbalanced"] = analysis["imbalance_ratio"] > 3

        if analysis["is_imbalanced"]:
            analysis["recommendation"] = "Consider class weights, oversampling (SMOTE), or undersampling"
    else:
        analysis["task_type"] = "regression"
        analysis["statistics"] = {
            "mean": target.mean(),
            "std": target.std(),
            "min": target.min(),
            "max": target.max(),
            "skewness": target.skew()
        }

        if abs(target.skew()) > 1:
            analysis["recommendation"] = "Target is skewed, consider log transform"

    return analysis
```

### 3. Data Quality Checks

```python
def check_data_quality(self) -> Dict[str, Any]:
    """Comprehensive data quality assessment."""
    quality = {
        "missing_data": self._check_missing(),
        "duplicates": self._check_duplicates(),
        "constant_features": self._check_constant(),
        "highly_correlated": self._check_correlation(),
        "potential_leakage": self._check_leakage(),
    }

    # Overall quality score
    issues = sum(1 for v in quality.values() if v.get("has_issues", False))
    quality["overall_score"] = max(0, 100 - issues * 20)
    quality["summary"] = "Good" if issues == 0 else f"{issues} quality issues found"

    return quality

def _check_missing(self) -> Dict:
    """Check missing data patterns."""
    missing = self.data.isna().sum()
    missing_cols = missing[missing > 0]

    return {
        "has_issues": len(missing_cols) > 0,
        "n_columns_with_missing": len(missing_cols),
        "total_missing_pct": self.data.isna().mean().mean() * 100,
        "columns": missing_cols.to_dict() if len(missing_cols) < 20 else "Too many to list",
        "pattern": self._missing_pattern() if len(missing_cols) > 0 else None
    }

def _missing_pattern(self) -> str:
    """Identify missing data pattern."""
    missing = self.data.isna()

    # Check if missing completely at random (MCAR)
    # Simple heuristic: check correlation between missingness
    missing_cols = missing.sum()[missing.sum() > 0].index.tolist()

    if len(missing_cols) <= 1:
        return "isolated"

    # Check if missing values correlate across columns
    missing_corr = missing[missing_cols].corr()
    high_corr = (missing_corr.abs() > 0.5).sum().sum() - len(missing_cols)

    if high_corr > 0:
        return "correlated (MAR/MNAR likely)"
    return "random (MCAR likely)"

def _check_duplicates(self) -> Dict:
    """Check for duplicate rows."""
    n_duplicates = self.data.duplicated().sum()
    return {
        "has_issues": n_duplicates > 0,
        "n_duplicates": n_duplicates,
        "duplicate_pct": n_duplicates / len(self.data) * 100
    }

def _check_constant(self) -> Dict:
    """Check for constant/near-constant features."""
    variance = self.data.var(numeric_only=True)
    constant = variance[variance == 0].index.tolist()

    # Near-constant: >99% same value
    near_constant = []
    for col in self.data.columns:
        if self.data[col].value_counts(normalize=True).iloc[0] > 0.99:
            if col not in constant:
                near_constant.append(col)

    return {
        "has_issues": len(constant) > 0 or len(near_constant) > 0,
        "constant_columns": constant,
        "near_constant_columns": near_constant
    }

def _check_correlation(self, threshold: float = 0.95) -> Dict:
    """Check for highly correlated features."""
    numeric = self.data.select_dtypes(include=[np.number])
    if len(numeric.columns) < 2:
        return {"has_issues": False, "pairs": []}

    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    high_corr_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            if upper.loc[idx, col] > threshold:
                high_corr_pairs.append((idx, col, upper.loc[idx, col]))

    return {
        "has_issues": len(high_corr_pairs) > 0,
        "pairs": high_corr_pairs,
        "recommendation": "Consider removing one feature from each highly correlated pair"
    }

def _check_leakage(self) -> Dict:
    """Check for potential data leakage indicators."""
    if self.target_col is None:
        return {"has_issues": False, "message": "No target specified"}

    leakage_suspects = []
    target = self.data[self.target_col]

    for col in self.data.columns:
        if col == self.target_col:
            continue

        # Check for suspiciously high correlation with target
        if self.data[col].dtype in [np.float64, np.int64]:
            corr = self.data[col].corr(target.astype(float) if target.dtype == 'object' else target)
            if abs(corr) > 0.95:
                leakage_suspects.append((col, f"correlation={corr:.3f}"))

        # Check if column name suggests future information
        suspicious_names = ['future', 'label', 'target', 'outcome', 'result']
        if any(s in col.lower() for s in suspicious_names):
            leakage_suspects.append((col, "suspicious name"))

    return {
        "has_issues": len(leakage_suspects) > 0,
        "suspects": leakage_suspects,
        "recommendation": "Manually verify these features don't contain future information"
    }
```

### 4. ML-Specific Analysis

```python
def ml_readiness_assessment(self) -> Dict[str, Any]:
    """Assess dataset readiness for ML training."""

    assessment = {
        "sample_size": self._assess_sample_size(),
        "feature_quality": self._assess_features(),
        "preprocessing_needed": self._suggest_preprocessing(),
        "recommended_models": self._recommend_models(),
    }

    return assessment

def _assess_sample_size(self) -> Dict:
    """Assess if sample size is adequate."""
    n_samples = len(self.data)
    n_features = len(self.data.columns) - (1 if self.target_col else 0)

    ratio = n_samples / max(n_features, 1)

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "samples_per_feature": ratio,
        "assessment": "adequate" if ratio > 10 else "low" if ratio > 5 else "very_low",
        "recommendation": None if ratio > 10 else "Consider feature selection or dimensionality reduction"
    }

def _suggest_preprocessing(self) -> List[str]:
    """Suggest preprocessing steps."""
    suggestions = []

    # Missing data handling
    if self.data.isna().any().any():
        missing_pct = self.data.isna().mean().mean() * 100
        if missing_pct < 5:
            suggestions.append("Impute missing values (mean/median/mode)")
        else:
            suggestions.append("Consider multiple imputation or specialized missing data handling")

    # Scaling
    numeric = self.data.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        ranges = numeric.max() - numeric.min()
        if ranges.std() > ranges.mean():
            suggestions.append("Standardize/normalize features (different scales detected)")

    # Categorical encoding
    categorical = self.data.select_dtypes(include=['object', 'category'])
    if len(categorical.columns) > 0:
        high_cardinality = [col for col in categorical.columns
                          if self.data[col].nunique() > 10]
        if high_cardinality:
            suggestions.append(f"Use target encoding for high-cardinality categoricals: {high_cardinality}")
        suggestions.append("Encode categorical variables (one-hot or ordinal)")

    # Skewness
    for col in numeric.columns:
        if abs(self.data[col].skew()) > 1:
            suggestions.append(f"Consider log transform for skewed feature: {col}")
            break

    return suggestions

def _recommend_models(self) -> Dict[str, List[str]]:
    """Recommend models based on dataset characteristics."""
    n_samples = len(self.data)
    n_features = len(self.data.columns)

    task_type = self.analyze_target().get("task_type", "unknown")

    recommendations = {"baseline": [], "recommended": [], "advanced": []}

    if task_type == "classification":
        recommendations["baseline"] = ["LogisticRegression", "DecisionTree"]
        if n_samples < 10000:
            recommendations["recommended"] = ["RandomForest", "GradientBoosting", "SVM"]
        else:
            recommendations["recommended"] = ["XGBoost", "LightGBM", "CatBoost"]
        recommendations["advanced"] = ["Neural Networks", "Ensemble methods"]

    elif task_type == "regression":
        recommendations["baseline"] = ["LinearRegression", "Ridge"]
        recommendations["recommended"] = ["RandomForestRegressor", "XGBoost", "LightGBM"]
        recommendations["advanced"] = ["Neural Networks", "Stacking ensemble"]

    return recommendations
```

### 5. Visualization Recommendations

```python
def get_visualization_plan(self) -> List[Dict[str, str]]:
    """Generate visualization recommendations."""
    plots = []

    # Target distribution
    if self.target_col:
        task_type = self.analyze_target().get("task_type", "unknown")
        if task_type == "classification":
            plots.append({
                "type": "bar",
                "title": "Class Distribution",
                "column": self.target_col,
                "code": f"df['{self.target_col}'].value_counts().plot(kind='bar')"
            })
        else:
            plots.append({
                "type": "histogram",
                "title": "Target Distribution",
                "column": self.target_col,
                "code": f"df['{self.target_col}'].hist(bins=50)"
            })

    # Correlation heatmap
    numeric = self.data.select_dtypes(include=[np.number])
    if len(numeric.columns) > 1:
        plots.append({
            "type": "heatmap",
            "title": "Feature Correlations",
            "code": "sns.heatmap(df.corr(), annot=True, cmap='coolwarm')"
        })

    # Missing data visualization
    if self.data.isna().any().any():
        plots.append({
            "type": "matrix",
            "title": "Missing Data Pattern",
            "code": "import missingno as msno; msno.matrix(df)"
        })

    return plots
```

## Output Format

Provide:
1. **Dataset summary** - Shape, types, memory usage
2. **Feature analysis** - Statistics for numerical/categorical features
3. **Target analysis** - Distribution, class balance, task type
4. **Data quality report** - Missing data, duplicates, leakage risks
5. **ML readiness assessment** - Sample size adequacy, preprocessing needs
6. **Preprocessing recommendations** - Specific steps for this dataset
7. **Visualization suggestions** - Key plots to generate
8. **Model recommendations** - Suitable algorithms for this data
