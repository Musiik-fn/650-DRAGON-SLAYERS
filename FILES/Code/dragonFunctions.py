import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, auc
)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats  # Added import for scipy.stats
from typing import List, Tuple
from sklearn.base import BaseEstimator

def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and confidence interval for a list of numbers.
    """
    a = np.array(data)
    n = len(a)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    mean = np.mean(a)
    se = stats.sem(a)
    if se == 0:
        return mean, (mean, mean)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, (mean - h, mean + h)

def get_stats(df, group_col='MORTALITY', confidence=0.95):
    """
    Analyze numerical features in the dataframe, aggregating by the specified group column.
    
    Parameters:
    - df: pandas DataFrame
    - group_col: column name to group by (binary: 0 or 1)
    - confidence: confidence level for intervals
    
    Returns:
    - result_df: pandas DataFrame with mean (CI) for each group and p-value
    """
    # Ensure the group column is binary
    if not set(df[group_col].unique()).issubset({0, 1}):
        raise ValueError(f"{group_col} must be binary (0 and 1).")
    
    # Select numerical columns excluding the group column
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != group_col]
    
    result = []
    
    for col in numerical_cols:
        group0 = df[df[group_col] == 0][col].dropna()
        group1 = df[df[group_col] == 1][col].dropna()
        
        # Calculate means and confidence intervals
        mean0, ci0 = mean_confidence_interval(group0, confidence)
        mean1, ci1 = mean_confidence_interval(group1, confidence)
        
        # Perform t-test
        if len(group0) < 2 or len(group1) < 2:
            p_value = np.nan
        else:
            try:
                t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=False, nan_policy='omit')
            except Exception:
                p_value = np.nan
        
        result.append({
            'Feature': col,
            f'Mean (0) [{int(confidence*100)}% CI]': f"{mean0:.4f} ({ci0[0]:.4f}, {ci0[1]:.4f})",
            f'Mean (1) [{int(confidence*100)}% CI]': f"{mean1:.4f} ({ci1[0]:.4f}, {ci1[1]:.4f})",
            'p-value': p_value
        })
    
    result_df = pd.DataFrame(result)
    return result_df

def preprocess_outliers(df, threshold=3):
    """
    Detects and caps outliers in non-binary numerical columns of the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - threshold (float): Number of standard deviations to define outliers (default is 3).
    
    Returns:
    - df_capped (pd.DataFrame): DataFrame with outliers capped.
    - summary_before (pd.DataFrame): Summary statistics before capping.
    - summary_after (pd.DataFrame): Summary statistics after capping.
    - capped_summary (pd.DataFrame): Count and percentage of capped values per column.
    - binary_numerical_cols (list): List of binary numerical columns.
    """
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Detect binary numerical columns
    binary_numerical_cols = []
    for col in numerical_cols:
        unique_values = df[col].dropna().unique()
        if len(unique_values) == 2:
            binary_numerical_cols.append(col)
    
    # Separate non-binary numerical columns
    non_binary_numerical_cols = [col for col in numerical_cols if col not in binary_numerical_cols]
    
    # Calculate mean and std dev
    mean_values = df[non_binary_numerical_cols].mean()
    std_values = df[non_binary_numerical_cols].std()
    
    # Define bounds
    lower_bound = mean_values - threshold * std_values
    upper_bound = mean_values + threshold * std_values
    
    # Detect outliers
    outliers = pd.DataFrame(False, index=df.index, columns=non_binary_numerical_cols)
    for col in non_binary_numerical_cols:
        outliers[col] = (df[col] < lower_bound[col]) | (df[col] > upper_bound[col])
    
    # Summary statistics before capping
    summary_before = df[non_binary_numerical_cols].describe().transpose()
    summary_before = summary_before[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    summary_before.columns = ['Count', 'Mean', 'Std_Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max']
    
    # Cap outliers
    df_capped = df.copy()
    for col in non_binary_numerical_cols:
        df_capped[col] = np.where(
            df_capped[col] < lower_bound[col],
            lower_bound[col],
            df_capped[col]
        )
        df_capped[col] = np.where(
            df_capped[col] > upper_bound[col],
            upper_bound[col],
            df_capped[col]
        )
    
    # Summary statistics after capping
    summary_after = df_capped[non_binary_numerical_cols].describe().transpose()
    summary_after = summary_after[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    summary_after.columns = ['Count', 'Mean_After', 'Std_Dev_After', 'Min_After', 'Q1_After', 'Median_After', 'Q3_After', 'Max_After']
    
    # Count of capped values per column
    capped_lower = (df[non_binary_numerical_cols] < lower_bound).sum()
    capped_upper = (df[non_binary_numerical_cols] > upper_bound).sum()
    total_capped = capped_lower + capped_upper
    total_entries = df[non_binary_numerical_cols].notna().sum()
    percent_capped = (total_capped / total_entries) * 100
    
    capped_summary = pd.DataFrame({
        'Capped_Lower': capped_lower,
        'Capped_Upper': capped_upper,
        'Total_Capped': total_capped,
        'Total_Entries': total_entries,
        'Percent_Capped': percent_capped.round(2)
    })
    
    # Overall totals
    overall_capped_lower = capped_lower.sum()
    overall_capped_upper = capped_upper.sum()
    overall_total_capped = overall_capped_lower + overall_capped_upper
    overall_total_entries = total_entries.sum()
    overall_percent_capped = (overall_total_capped / overall_total_entries) * 100
    
    overall_summary = pd.DataFrame({
        'Capped_Lower': [overall_capped_lower],
        'Capped_Upper': [overall_capped_upper],
        'Total_Capped': [overall_total_capped],
        'Total_Entries': [overall_total_entries],
        'Percent_Capped': [round(overall_percent_capped, 2)]
    }, index=['Overall'])
    
    # Append overall summary to capped_summary
    capped_summary = pd.concat([capped_summary, overall_summary])
    
    return df_capped, summary_before, summary_after, capped_summary, binary_numerical_cols

def compare_summary_stats(summary_before, summary_after, capped_summary):
    """
    Compares summary statistics before and after outlier handling.
    
    Parameters:
    - summary_before (pd.DataFrame): Summary statistics before handling outliers.
    - summary_after (pd.DataFrame): Summary statistics after handling outliers.
    - capped_summary (pd.DataFrame): Count and percentage of capped values per column.
    
    Returns:
    - comparison_table (pd.DataFrame): Table showing before, after, changes, and outlier counts.
    """
    # Initialize comparison DataFrame
    comparison_table = pd.DataFrame(index=summary_before.index)
    
    # Mean
    comparison_table['Mean Before'] = summary_before['Mean']
    comparison_table['Mean After'] = summary_after['Mean_After']
    comparison_table['Mean Change'] = comparison_table['Mean After'] - comparison_table['Mean Before']
    comparison_table['Mean % Change'] = (comparison_table['Mean Change'] / comparison_table['Mean Before']) * 100
    
    # Std Dev
    comparison_table['Std Dev Before'] = summary_before['Std_Dev']
    comparison_table['Std Dev After'] = summary_after['Std_Dev_After']
    comparison_table['Std Dev Change'] = comparison_table['Std Dev After'] - summary_before['Std_Dev']
    comparison_table['Std Dev % Change'] = (comparison_table['Std Dev Change'] / comparison_table['Std Dev Before']) * 100
    
    # Min
    comparison_table['Min Before'] = summary_before['Min']
    comparison_table['Min After'] = summary_after['Min_After']
    comparison_table['Min Change'] = comparison_table['Min After'] - summary_before['Min']
    comparison_table['Min % Change'] = (comparison_table['Min Change'] / summary_before['Min']) * 100
    
    # Max
    comparison_table['Max Before'] = summary_before['Max']
    comparison_table['Max After'] = summary_after['Max_After']
    comparison_table['Max Change'] = comparison_table['Max After'] - summary_before['Max']
    comparison_table['Max % Change'] = (comparison_table['Max Change'] / summary_before['Max']) * 100
    
    # Outlier Counts
    comparison_table['Capped_Lower'] = capped_summary['Capped_Lower']
    comparison_table['Capped_Upper'] = capped_summary['Capped_Upper']
    comparison_table['Total_Capped'] = capped_summary['Total_Capped']
    comparison_table['Percent_Capped'] = capped_summary['Percent_Capped']
    
    # Arrange columns for better readability
    comparison_table = comparison_table[[
        'Mean Before', 'Mean After', 'Mean Change', 'Mean % Change',
        'Std Dev Before', 'Std Dev After', 'Std Dev Change', 'Std Dev % Change',
        'Min Before', 'Min After', 'Min Change', 'Min % Change',
        'Max Before', 'Max After', 'Max Change', 'Max % Change',
        'Capped_Lower', 'Capped_Upper', 'Total_Capped', 'Percent_Capped'
    ]]
    
    return comparison_table

def save_dataframe(df, filename, directory):
    """
    Saves a DataFrame to a specified directory with the given filename.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - filename (str): The name of the file (e.g., 'RandomForestImportance.csv').
    - directory (Path): The directory path where the file will be saved.
    """
    filepath = directory / filename
    df.to_csv(filepath, index=False)
    print(f"DataFrame saved successfully at {filepath.resolve()}")

# Define a function to save plots
def save_plot(plot_path):
    """
    Saves the current matplotlib plot to the specified path.

    Parameters:
    - plot_path (Path): The full path (including filename) where the plot will be saved.
    """
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully at {plot_path.resolve()}")
  
def evaluate_model(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    color,
    data_directory,
    plot_directory,
    kf
):
    """
    Trains the model, performs cross-validation, plots and saves confusion matrix,
    extracts and plots feature importances, and evaluates on the test set with plots saved.

    Parameters:
    - model: The machine learning model to evaluate.
    - model_name (str): Name of the model.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training labels.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing labels.
    - color (str): Color for the model's plots.
    - data_directory (Path): Directory to save CSV files.
    - plot_directory (Path): Directory to save plot PNGs.
    - kf (KFold): Cross-validation strategy.

    Returns:
    - metrics (dict): Dictionary containing ROC and PR metrics.
    """
    print(f"\n--- Evaluating {model_name} ---")
    
    # Cross-Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    print(f'{model_name} Cross-Validation Accuracy Scores: {cv_scores}')
    print(f'{model_name} Average Cross-Validation Accuracy: {cv_scores.mean():.4f}')
    
    # Cross-Validated Predictions
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=kf)
    cm_cv = confusion_matrix(y_train, y_pred_cv)
    
    # Plot Confusion Matrix (Cross-Validated Training)
    plt.figure(figsize=(8, 6))
    if model_name == 'Random Forest':
        cmap = 'Greens'
    elif model_name == 'Logistic Regression':
        cmap = 'Blues'
    elif model_name == 'XGBoost':
        cmap = 'Oranges'
    else:
        cmap = 'viridis'
    
    sns.heatmap(
        cm_cv, annot=True, fmt='d', cmap=cmap,
        xticklabels=['Did Not Survive', 'Survived'],
        yticklabels=['Did Not Survive', 'Survived'],
        cbar=False
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model_name} (Cross-Validated Training)')
    
    # Save Confusion Matrix Plot
    cm_plot_path = plot_directory / f'Confusion_Matrix_{model_name}_Cross_Validated_Training.png'
    save_plot(cm_plot_path)
    plt.show()
    plt.close()
    
    # Fit on Entire Training Data
    model.fit(X_train, y_train)
    
    # Feature Importances or Coefficients
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(15)
        
        # Save feature importances CSV
        save_dataframe(feature_importances_df, f'{model_name}FeatureImportances.csv', data_directory)
        
        # Plot Feature Importances
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importances_df,
            palette=cmap
        )
        plt.title(f'Top 15 Feature Importances from {model_name}')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        
        # Save Feature Importances Plot
        fi_plot_path = plot_directory / f'Feature_Importances_{model_name}.png'
        save_plot(fi_plot_path)
        plt.show()
        plt.close()
        
    elif hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        coefficients_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': coefficients
        }).assign(Abs_Coefficient=lambda df: df['Coefficient'].abs()).sort_values(by='Abs_Coefficient', ascending=False).head(15)
        
        # Save coefficients CSV
        save_dataframe(coefficients_df, f'{model_name}Coefficients.csv', data_directory)
        
        # Plot Coefficients
        plt.figure(figsize=(10, 8))
        sns.barplot(
            x='Coefficient',
            y='Feature',
            data=coefficients_df,
            palette=cmap
        )
        plt.title(f'Top 15 Feature Coefficients from {model_name}')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        
        # Save Coefficients Plot
        coef_plot_path = plot_directory / f'Feature_Coefficients_{model_name}.png'
        save_plot(coef_plot_path)
        plt.show()
        plt.close()
    
    # Test Set Predictions
    y_test_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_test_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Use decision function if predict_proba not available
        y_test_prob = model.decision_function(X_test)
    
    # Classification Report
    print(f"Classification Report - {model_name} (Test Set)")
    print(classification_report(y_test, y_test_pred))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_test_prob)
    print(f'{model_name} ROC-AUC Score on Test Set: {roc_auc:.4f}')
    
    # Compute ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr,
        label=f'{model_name} (AUC = {roc_auc:.4f})',
        color=color,
        lw=2
    )
    plt.plot([0, 1], [0, 1], 'k--', color='grey', lw=2)  # Diagonal line
    plt.fill_between(fpr, tpr, alpha=0.3, color=color)  # Shade the area under the ROC curve
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # Save ROC Curve Plot
    roc_plot_path = plot_directory / f'ROC_Curve_{model_name}.png'
    save_plot(roc_plot_path)
    plt.show()
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    average_precision = average_precision_score(y_test, y_test_prob)
    
    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision,
        color=color,
        lw=2,
        label=f'PR curve (AP = {average_precision:.4f})'
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall (PR) Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    # Save Precision-Recall Curve Plot
    pr_plot_path = plot_directory / f'Precision_Recall_Curve_{model_name}.png'
    save_plot(pr_plot_path)
    plt.show()
    plt.close()
    
    # Collect Metrics for Combined Plotting
    metrics = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'average_precision': average_precision
    }
    
    return metrics

def plot_combined_roc(model_metrics, plot_directory, model_colors):
    """
    Plots ROC curves for all models on a single plot.

    Parameters:
    - model_metrics (dict): Dictionary containing ROC metrics for each model.
    - plot_directory (Path): Directory to save the combined ROC plot.
    - model_colors (dict): Dictionary mapping model names to colors.
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in model_metrics.items():
        plt.plot(
            metrics['fpr'],
            metrics['tpr'],
            label=f"{model_name} (AUC = {metrics['roc_auc']:.4f})",
            lw=2,
            color=model_colors[model_name]
        )
    
    # Diagonal line for random guessing
    plt.plot([0, 1], [0, 1], 'k--', color='grey', lw=2)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # Save Combined ROC Plot
    combined_roc_path = plot_directory / 'Combined_ROC_Curves.png'
    save_plot(combined_roc_path)
    plt.show()
    plt.close()
    
    print(f"Combined ROC Curves plot saved at {combined_roc_path.resolve()}")

def plot_combined_pr(model_metrics, plot_directory, model_colors):
    """
    Plots Precision-Recall curves for all models on a single plot.

    Parameters:
    - model_metrics (dict): Dictionary containing PR metrics for each model.
    - plot_directory (Path): Directory to save the combined PR plot.
    - model_colors (dict): Dictionary mapping model names to colors.
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in model_metrics.items():
        plt.plot(
            metrics['recall'],
            metrics['precision'],
            label=f"{model_name} (AP = {metrics['average_precision']:.4f})",
            lw=2,
            color=model_colors[model_name]
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curves Comparison')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.show()
    # Save Combined PR Plot
    combined_pr_path = plot_directory / 'Combined_Precision_Recall_Curves.png'
    save_plot(combined_pr_path)
    
    plt.close()
    
    print(f"Combined Precision-Recall Curves plot saved at {combined_pr_path.resolve()}")

def clinical_impact_curve(
    probabilities: np.ndarray, 
    outcomes: np.ndarray, 
    thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the number of high-risk patients and true positives at each threshold.

    Parameters:
    ----------
    probabilities : np.ndarray
        Predicted probabilities for the positive class.
    outcomes : np.ndarray
        True binary outcomes.
    thresholds : np.ndarray
        Array of threshold values.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        high_risk: Number of high-risk patients at each threshold.
        true_positives: Number of true positives at each threshold.
    """
    high_risk = (probabilities[:, None] >= thresholds).sum(axis=0)
    true_positives = ((probabilities[:, None] >= thresholds) & (outcomes[:, None] == 1)).sum(axis=0)
    return high_risk, true_positives

def bootstrap_confidence_intervals(
    probabilities: np.ndarray, 
    outcomes: np.ndarray, 
    thresholds: np.ndarray, 
    n_bootstraps: int = 1000, 
    ci: float = 95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for high-risk and true positive counts.

    Parameters:
    ----------
    probabilities : np.ndarray
        Predicted probabilities for the positive class.
    outcomes : np.ndarray
        True binary outcomes.
    thresholds : np.ndarray
        Array of threshold values.
    n_bootstraps : int, optional
        Number of bootstrap samples, by default 1000.
    ci : float, optional
        Confidence interval percentage, by default 95.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        high_risk_lower, high_risk_upper, true_positive_lower, true_positive_upper
    """
    rng = np.random.default_rng()
    high_risk_boot = np.empty((n_bootstraps, len(thresholds)), dtype=int)
    true_pos_boot = np.empty((n_bootstraps, len(thresholds)), dtype=int)

    for i in range(n_bootstraps):
        indices = rng.integers(0, len(probabilities), len(probabilities))
        prob_sample = probabilities[indices]
        outcome_sample = outcomes[indices]
        hr, tp = clinical_impact_curve(prob_sample, outcome_sample, thresholds)
        high_risk_boot[i, :] = hr
        true_pos_boot[i, :] = tp

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    high_risk_lower = np.percentile(high_risk_boot, lower_percentile, axis=0)
    high_risk_upper = np.percentile(high_risk_boot, upper_percentile, axis=0)
    true_positive_lower = np.percentile(true_pos_boot, lower_percentile, axis=0)
    true_positive_upper = np.percentile(true_pos_boot, upper_percentile, axis=0)

    return high_risk_lower, high_risk_upper, true_positive_lower, true_positive_upper


def plot_clinical_impact_curve(
    model: BaseEstimator, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    n_bootstraps: int = 1000, 
    ci: float = 95
) -> None:
    """
    Plot the clinical impact curve for a given model and test data, including 95% confidence intervals.

    Parameters:
    ----------
    model : BaseEstimator
        Trained machine learning model with `predict_proba` method.
    X_test : pd.DataFrame
        Test data.
    y_test : pd.Series or np.ndarray
        True binary outcomes for the test data.
    n_bootstraps : int, optional
        Number of bootstrap samples for CI estimation, by default 1000.
    ci : float, optional
        Confidence interval percentage, by default 95.
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError("The model does not have a `predict_proba` method.")

    thresholds = np.linspace(0.0, 1.0, 100)
    probabilities = model.predict_proba(X_test)[:, 1]
    outcomes = np.array(y_test)

    # Calculate point estimates
    high_risk, true_positives = clinical_impact_curve(probabilities, outcomes, thresholds)

    # Calculate confidence intervals via bootstrapping
    high_risk_lower, high_risk_upper, true_positive_lower, true_positive_upper = bootstrap_confidence_intervals(
        probabilities, outcomes, thresholds, n_bootstraps=n_bootstraps, ci=ci
    )

    # Plotting the clinical impact curve
    plt.figure(figsize=(10, 6))
    
    # High-risk patients
    plt.plot(thresholds, high_risk, label='Number High Risk', color='red', linestyle='--')
    plt.fill_between(thresholds, high_risk_lower, high_risk_upper, color='red', alpha=0.2, label=f'{ci}% CI High Risk')
    
    # True positives
    plt.plot(thresholds, true_positives, label='Number of True Positives', color='blue', linestyle='-')
    plt.fill_between(thresholds, true_positive_lower, true_positive_upper, color='blue', alpha=0.2, label=f'{ci}% CI True Positives')
    
    plt.xlabel('Threshold Probability')
    plt.ylabel('Number of Patients')
    plt.title('Clinical Impact Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from sklearn.calibration import calibration_curve

def plot_calibration_curve(model, X_test, y_test, model_name):
    """
    Plots the calibration curve for a given model.
    
    Parameters:
    - model: Trained model with predict_proba method.
    - X_test: Test features.
    - y_test: Test labels.
    - model_name: Name of the model.
    - plot_directory: Directory to save the plot.
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model does not have predict_proba method.")
    
    # Get predicted probabilities
    prob_pos = model.predict_proba(X_test)[:, 1]
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
    
    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)