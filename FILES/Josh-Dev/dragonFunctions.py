import pandas as pd
from scipy import stats
import numpy as np

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

    