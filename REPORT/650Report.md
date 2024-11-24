# 650 Project - Predicting 30‑days mortality for MIMIC‑III patients with sepsis‑3

**Authors**: Joshua Cabal,  Ida Karima, Viridiana Radillo,

California State University, Northridge

December 4, 2024

---

Table of Contents:
- [650 Project - Predicting 30‑days mortality for MIMIC‑III patients with sepsis‑3](#650-project---predicting-30days-mortality-for-mimiciii-patients-with-sepsis3)
  - [Executive Sumamry](#executive-sumamry)
  - [Introduction](#introduction)
    - [Background](#background)
    - [Challenges in Predicting Sepsis Outcomes](#challenges-in-predicting-sepsis-outcomes)
    - [Purpose of the Study](#purpose-of-the-study)
    - [Objectives](#objectives)
    - [Significance](#significance)
  - [Methodology](#methodology)
    - [Database Interfacing](#database-interfacing)
    - [XGBoost](#xgboost)
  - [Data Extraction](#data-extraction)
    - [Patient Selection Criteria and Query](#patient-selection-criteria-and-query)
    - [Feature Selection](#feature-selection)
    - [Data Aggregation](#data-aggregation)
  - [Data Preprocessing](#data-preprocessing)
    - [Defining Outcome Variable: 30-Day Mortality](#defining-outcome-variable-30-day-mortality)
    - [Categorical Feature Encoding](#categorical-feature-encoding)
    - [Handling `NULL` Values](#handling-null-values)
    - [Handling Outliers](#handling-outliers)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Patient Cohort](#patient-cohort)
    - [Feature Statistics](#feature-statistics)
  - [Model Development and Evaluation](#model-development-and-evaluation)
  - [Discussion](#discussion)
  - [Conclusion](#conclusion)
  - [References](#references)

---

## Executive Sumamry

Sepsis is a critical global health issue with high mortality rates, particularly among ICU patients. Early prediction of sepsis outcomes is essential for improving patient care and survival rates. This report replicates the study by [Hou et al. (2020)](https://doi.org/10.1186/s12967-020-02620-5), aiming to **develop a predictive model for 30-day mortality in sepsis-3 patients** using the MIMIC-III database.

Using machine learning techniques, specifically the XGBoost algorithm, we constructed a predictive model and compared its performance with traditional models such as random forests and logistic regression. Our findings indicate that the XGBoost model outperforms the conventional models, demonstrating higher accuracy and better predictive capabilities.

This study reinforces the potential of machine learning approaches in clinical settings, suggesting that the XGBoost model could assist clinicians in making informed decisions and tailoring precise treatments for sepsis patients.

## Introduction

### Background

Sepsis, a life-threatening organ dysfunction caused by a dysregulated host response to infection, remains a leading cause of mortality in intensive care units (ICUs) worldwide. With over [5 million deaths annually](https://www.who.int/news-room/fact-sheets/detail/sepsis), the burden of sepsis on healthcare systems is profound, necessitating improved strategies for early detection and management.

### Challenges in Predicting Sepsis Outcomes

Traditional biochemical markers such as white blood cell count, C-reactive protein (CRP), and procalcitonin (PCT) lack the sensitivity and specificity needed to reliably identify infections, leading to inconsistent clinical assessments and excessive use of broad-spectrum antibiotics ([Duncan et al. 2021](https://doi.org/10.1007/s11908-021-00765-y)). Novel approaches, such as machine learning models that integrate multiple clinical and diagnostic inputs, as well as gene expression profiling, show promise in improving diagnostic accuracy but remain largely experimental

### Purpose of the Study

This report aims to replicate the study conducted by [Hou et al. (2020)](https://doi.org/10.1186/s12967-020-02620-5), which developed an XGBoost-based machine learning model to predict 30-day mortality in sepsis-3 patients using data from the MIMIC-III database. By replicating this study, we seek to validate their findings and assess the model's applicability in predicting sepsis outcomes.

### Objectives

- To develop a machine learning model using XGBoost for predicting 30-day mortality in sepsis-3 patients.
- To compare the performance of the XGBoost model with traditional models such as logstic regression and random forests.
- To evaluate the clinical utility of the XGBoost model through validation techniques.
- To compare and contrast our process and findings with the replicated paper.

### Significance
Replicating Hou et al.'s study is crucial for substantiating the initial promising findings that XGBoost can enhance mortality prediction in sepsis-3 patients. Robust and validated machine learning models have the potential to significantly impact patient care by enabling timely and personalized interventions, thereby reducing mortality rates associated with sepsis.

Validating and potentially enhancing predictive models for sepsis outcomes can aid clinicians in early identification of high-risk patients, enabling timely interventions and personalized treatment strategies. This study contributes to the growing body of evidence supporting the integration of machine learning in critical care settings.

## Methodology

### Database Interfacing

Similar to the source paper, we used the Medical Information Mart for Intensive Care III database version 1.4 ([MIMIC III v1.4](https://doi.org/10.1038/sdata.2016.35)) for the study. MIMIC-III, a publicly available single-center critical care database which was approved by the Institutional Review Boards of Beth Israel Deaconess Medical Center (BIDMC, Boston, MA, USA) and the Massachusetts Institute of Technology (MIT, Cambridge, MA, USA), includes information on 46,520 patients who were admitted to various ICUs of BIDMC in Boston, Massachusetts from 2001 to 2012. The data was accessed via the Google BigQuery cloud platform and subsequently extracted and processed using SQL and Python. Additionally, the models were developed and evaluated in Python. This approach contrasts with the target paper, which primarily utilized R for these tasks.

### XGBoost

From the [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/),  XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.


## Data Extraction

Below is an image of the process which was used to prepare the data for model development. 

![Patient Cohort Image](Report%20Figures/Data%20Preprocessing%20Figure(1).png)

We first extracted the given admission and patient data of all patients who were diagnosed with Sepsis. After this, we extracted the related baseline, vital, and laboratory data related to each sepsis patient. All variables were loaded into their own dataframe and eventually all dataframes were merged onto the patient list which was initially extracted. After this, the data transformation steps included handling inconsistencies, missing values, outliers, and categorical feature encoding. Finally, the 30 day mortality column is defined.

All SQL statements used can be found in the file `Data_Extraction.sql`.

### Patient Selection Criteria and Query

The following criteria were used in the selection of the patient records:

- Patient must be diagnosed with sepsis. ICD9 Codes:
    - 99591, Sepsis
    - 99592, Severe sepsis
- Patient must be at least 18 years old
- Patient must have demographic data
- Patient must have related lab test results
- Patient must have no more than 20% of data missing

Below is our baseline patient query. The criteria mentioned above were applied after the dataframe merge was completed.
```SQL
SELECT DISTINCT
    p.SUBJECT_ID,
    p.GENDER,
    p.DOB,
    p.DOD,
    p.EXPIRE_FLAG,
    a.HADM_ID,
    a.ADMITTIME,
    a.DISCHTIME,
    a.ADMISSION_TYPE,
    DATE_DIFF(DATE(a.ADMITTIME), DATE(p.DOB), YEAR) AS AGE_AT_ADMISSION
FROM
    physionet-data.mimiciii_clinical.patients p
JOIN
    physionet-data.mimiciii_clinical.diagnoses_icd d
    ON p.SUBJECT_ID = d.SUBJECT_ID
JOIN
    physionet-data.mimiciii_clinical.admissions a
    ON d.HADM_ID = a.HADM_ID
WHERE
    d.ICD9_CODE IN ('99591', '99592');
```

### Feature Selection

The features were divided into the following categories: **baseline variables, vital signs, and laboratory parameters**. The variables were chosen directly from the paper. For the variables that were captured periodically, such as heart rate, they were aggregated and incorporated into the model using **min**, **mean**, and **maximum values**. 


| Feature Type          | Feature                          | Source Table | Item ID                                                         |
|-----------------------|----------------------------------|--------------|-----------------------------------------------------------------|
| Baseline Variables    | Age (year)                       | PATIENTS     | Given                                                           |
| Baseline Variables    | Sex                              | PATIENTS     | Given                                                           |
| Baseline Variables    | Ethnicity                        | PATIENTS     | Given                                                           |
| Baseline Variables    | Weight (kg)                      | CHARTEVENTS  | 226531, 763, 224639, 226512                                      |
| Baseline Variables    | Height (cm)                      | CHARTEVENTS  | 226707, 226730, 1394                                             |
| Baseline Variables    | Length of stay in hospital (days)| ADMISSION    | Calculated                                                      |
| Baseline Variables    | Length of stay in ICU (days)      | ICUSTAYS     | Given                                                           |
| Vital Signs           | Heartrate_min (times/min)         | CHARTEVENTS  | 211, 220045                                                     |
| Vital Signs           | Heartrate_mean (times/min)        | CHARTEVENTS  | 211, 220045                                                     |
| Vital Signs           | Sysbp_min (mmHg)                  | CHARTEVENTS  | 51, 422, 227243, 224167, 220179, 225309, 6701, 220050, 455      |
| Vital Signs           | Diasbp_mean (mmHg)                | CHARTEVENTS  | 224643, 225310, 220180, 8555, 220051, 8368, 8441, 8440          |
| Vital Signs           | Meanbp_min (mmHg)                 | CHARTEVENTS  | 456, 220181, 224, 225312, 220052, 52, 6702, 224322             |
| Vital Signs           | Resprate_mean (times/min)         | CHARTEVENTS  | 224422, 618, 220210, 224689, 614, 651, 224690, 615             |
| Vital Signs           | Tempc_min (°C)                    | CHARTEVENTS  | 223761, 677, 676, 679, 678, 223762                              |
| Vital Signs           | Tempc_max (°C)                    | CHARTEVENTS  | 223761, 677, 676, 679, 678, 223762                              |
| Vital Signs           | Spo2_mean (%)                     | CHARTEVENTS  | 646, 50817, 834, 220277, 220227                                  |
| Laboratory Parameters | Aniongap_max (mmHg)               | LABEVENTS    | 50868                                                           |
| Laboratory Parameters | Aniongap_min (mmHg)               | LABEVENTS    | 50868                                                           |
| Laboratory Parameters | Creatinine_min (ng/dL)            | LABEVENTS    | 50912                                                           |
| Laboratory Parameters | Chloride_min (mmol/L)             | LABEVENTS    | 50806, 50902                                                     |
| Laboratory Parameters | Hemoglobin_min (g/dL)             | LABEVENTS    | 51222, 50811                                                     |
| Laboratory Parameters | Hemoglobin_max (g/dL)             | LABEVENTS    | 51222, 50811                                                     |
| Laboratory Parameters | Lactate_min (mmol/L)              | LABEVENTS    | 50813                                                           |
| Laboratory Parameters | Platelet_min (10⁹/L)              | LABEVENTS    | 51265                                                           |
| Laboratory Parameters | Potassium_min (mmol/L)            | LABEVENTS    | 50971, 50822                                                     |
| Laboratory Parameters | Sodium_min (mmol/L)               | LABEVENTS    | 50983, 50824                                                     |
| Laboratory Parameters | Sodium_max (mmol/L)               | LABEVENTS    | 50983, 50824                                                     |
| Laboratory Parameters | Bun_min (mmol/L)                  | LABEVENTS    | 51006                                                           |
| Laboratory Parameters | Bun_max (mmol/L)                  | LABEVENTS    | 51006                                                           |
| Laboratory Parameters | Wbc_min (10⁹/L)                    | LABEVENTS    | 51516                                                           |
| Laboratory Parameters | Wbc_max (10⁹/L)                    | LABEVENTS    | 51516                                                           |
| Laboratory Parameters | Inr_max                           | LABEVENTS    | 51237                                                           |

### Data Aggregation

Because many of the features are captured periodically, such as vital signs and lab events, these features are incorporated using their minimum, maximum, and mean.

Below is the heart rate query which was used. All queries which needed aggregation followed this essential format. 
```SQL
SELECT C.SUBJECT_ID,ITEM.LABEL, AVG(C.VALUENUM) AS HEARTRATE
FROM `physionet-data.mimiciii_clinical.chartevents` AS C

INNER JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` AS D ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN `physionet-data.mimiciii_clinical.d_items` AS ITEM ON ITEM.ITEMID = C.ITEMID
WHERE D.ICD9_CODE IN ('99591', '99592') AND (C.ITEMID  IN (211, 220045))

GROUP BY C.SUBJECT_ID, ITEM.LABEL
ORDER BY C.SUBJECT_ID ASC
```
A query was written to extract each feature listed in the feature table, and then the results were merged on the sepsis patient table on the `SUBJECT_ID`. 

## Data Preprocessing

### Defining Outcome Variable: 30-Day Mortality
The sepsis patients were divided into two groups based on their 30 day mortality:
1. `MORTALITY = 0`: Died within 30 days
2. `MORTALITY = 1`: Survived within 30 days

```python
patient_df['DOD'] = pd.to_datetime(patient_df['DOD'])
patient_df['ADMITTIME'] = pd.to_datetime(patient_df['ADMITTIME'])

# define 30-days mortality column 
patient_df['DIFF_DAYS'] = (patient_df['DOD'] - patient_df['ADMITTIME']).dt.days
patient_df['MORTALITY'] = patient_df['DIFF_DAYS'].apply(lambda x: 0 if x>30 else 1)
```

The statistics related to this will be provided in the exploratory data analysis section of this paper. 


### Categorical Feature Encoding

The following categorical features were used in the model: Sex, Ethnicity, and Admission Type. All features were encoded into binary variables. Originally, 36 unique ethnicities were reported, and because of this high cardinality, the following grouping was used: 

| Ethnicity Conditions                                                                 | Consolidation                         |
|------------------------------------------------------------------------------------|--------------------------------------------|
| Contains `'WHITE'`                                                                  | WHITE                                      |
| Contains `'BLACK'` or `'AFRICAN AMERICAN'`                                         | BLACK OR AFRICAN AMERICAN                  |
| Contains `'ASIAN'`                                                                  | ASIAN                                      |
| Contains `'HISPANIC'` or `'LATINO'`                                                 | HISPANIC OR LATINO                         |
| Contains `'AMERICAN INDIAN'` or `'ALASKA NATIVE'`                                   | AMERICAN INDIAN OR ALASKA NATIVE           |
| Contains `'NATIVE HAWAIIAN'` or `'PACIFIC ISLANDER'`                                | NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER   |
| Contains `'MIDDLE EASTERN'`                                                         | MIDDLE EASTERN                             |
| Contains `'UNKNOWN'`, `'NOT SPECIFIED'`, `'DECLINED TO ANSWER'`, or `'UNABLE TO OBTAIN'` | UNKNOWN/NOT SPECIFIED/DECLINED             |
| Does not meet any of the above conditions                                         | OTHER                                      |



### Handling `NULL` Values
After extracting all data and merging onto one DataFrame, a report of the `null` values was ran:

| FEATURE                                                    | NULL COUNT | NULL PERCENTAGE |
|------------------------------------------------------------|------------|------------------|
| AGE_AT_ADMISSION                                           | 0          | 0.000            |
| LOS                                                        | 0          | 0.000            |
| LOS_ICU_MEAN                                               | 1          | 0.022            |
| WEIGHT_MEAN                                                | 232        | 5.093            |
| HEARTRATE_MEAN                                             | 5          | 0.110            |
| SBP_MEAN                                                   | 4          | 0.088            |
| DBP_MEAN                                                   | 4          | 0.088            |
| MAP_MEAN                                                   | 4          | 0.088            |
| RR_MEAN                                                    | 4          | 0.088            |
| TEMP_MIN_C                                                 | 12         | 0.263            |
| TEMP_MAX_C                                                 | 12         | 0.263            |
| OXYGEN_SAT_MEAN                                            | 8          | 0.176            |
| DIABETES                                                   | 0          | 0.000            |
| ANIONGAP_MAX_VAL                                           | 4          | 0.088            |
| BUN_MAX_VAL                                                | 2          | 0.044            |
| HEMOGLOBIN_MAX_VAL                                         | 2          | 0.044            |
| INR_MAX_VAL                                                | 35         | 0.768            |
| SODIUM_MAX_VAL                                             | 3          | 0.066            |
| ANIONGAP_MIN_VAL                                           | 4          | 0.088            |
| BUN_MIN_VAL                                                | 2          | 0.044            |
| CHLORIDE_MIN_VAL                                           | 3          | 0.066            |
| CREATININE_MIN_VAL                                         | 2          | 0.044            |
| HEMOGLOBIN_MIN_VAL                                         | 2          | 0.044            |
| INR_MIN_VAL                                                | 35         | 0.768            |
| LACTATE_MIN_VAL                                            | 89         | 1.954            |
| PLATELET_MIN_VAL                                           | 2          | 0.044            |
| POTASSIUM_MIN_VAL                                          | 2          | 0.044            |
| SODIUM_MIN_VAL                                             | 3          | 0.066            |
| MORTALITY                                                  | 0          | 0.000            |
| GENDER_M                                                   | 0          | 0.000            |
| ADMISSION_TYPE_EMERGENCY                                    | 0          | 0.000            |
| ADMISSION_TYPE_URGENT                                       | 0          | 0.000            |
| ETHNICITY_CONSOLIDATED_ASIAN                                | 0          | 0.000            |
| ETHNICITY_CONSOLIDATED_BLACK OR AFRICAN AMERICAN            | 0          | 0.000            |
| ETHNICITY_CONSOLIDATED_HISPANIC OR LATINO                    | 0          | 0.000            |
| ETHNICITY_CONSOLIDATED_MIDDLE EASTERN                        | 0          | 0.000            |
| ETHNICITY_CONSOLIDATED_NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER | 0          | 0.000            |
| ETHNICITY_CONSOLIDATED_OTHER                                  | 0          | 0.000            |
| ETHNICITY_CONSOLIDATED_UNKNOWN/NOT SPECIFIED/DECLINED         | 0          | 0.000            |
| ETHNICITY_CONSOLIDATED_WHITE                                   | 0          | 0.000            |

Mean value imputation was used to fill all `null` values. 

### Handling Outliers

The outliers were handled using [winsorization](https://www.sciencedirect.com/science/article/abs/pii/B9780123848642000287). Through this method, extreme values were limited to a chosen distance of 3 standard deviations from the mean. The function which handled this is defined in the file `dragonFunctions.py`. The function also tracks statsitcs of the variables before and after the operation. The results are contained in the file named `Outlier_Report.csv`. Below is the docstring of the function as well as a portion of the outlier report. 

```python
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
```
| Feature Name        | Mean Before | Mean After | Mean % Change   | Capped_Lower | Capped_Upper | Total_Capped | Percent_Capped |
|---------------------|-------------|------------|-----------------|--------------|--------------|--------------|-----------------|
| LOS                 | 14.48222566 | 13.4672812 | -7.008207746    | 0            | 201          | 201          | 3.88            |
| LOS_ICU_MEAN        | 6.543660917 | 6.038849931| -7.714504042    | 0            | 224          | 224          | 4.33            |
| WEIGHT_MEAN         | 82.99531556 | 81.93668715| -1.275527906    | 8            | 185          | 193          | 3.73            |
| HEARTRATE_MEAN      | 89.05715229 | 88.91809416| -0.1561448274   | 78           | 153          | 231          | 4.46            |
| SBP_MEAN            | 114.0404809 | 113.5807231| -0.4031531588   | 9            | 8            | 17           | 0.33            |
| DBP_MEAN            | 59.20096918 | 58.85227332| -0.5890036279   | 15           | 65           | 80           | 1.55            |
| MAP_MEAN            | 75.40752183 | 75.29132781| -0.1540881075   | 80           | 161          | 241          | 4.66            |
| RR_MEAN             | 20.37821362 | 20.30474214| -0.3605393443   | 44           | 182          | 226          | 4.37            |
| TEMP_MEAN_C         | 36.83228947 | 36.83486969| 0.007005326655   | 29           | 19           | 48           | 0.93            |
| TEMP_MIN_C          | 33.84836104 | 34.51626202| 1.973215108      | 205          | 0            | 205          | 3.96            |
| TEMP_MAX_C          | 40.81430734 | 39.54394385| -3.112544525     | 0            | 62           | 62           | 1.2             |
| OXYGEN_SAT_MEAN     | 96.59966573 | 96.68731323| 0.09073272381     | 81           | 10           | 91           | 1.76            |
| ANIONGAP_MAX_VAL    | 21.49323279 | 21.26224981| -1.074677723      | 2            | 231          | 233          | 4.5             |
| BUN_MAX_VAL         | 61.55682257 | 60.55038318| -1.634976199      | 0            | 235          | 235          | 4.54            |
| CHLORIDE_MAX_VAL    | 112.8416779 | 112.735994 | -0.09365683408    | 116          | 151          | 267          | 5.16            |
| CREATININE_MAX_VAL  | 3.330228063 | 3.178131668| -4.567146543       | 0            | 8            | 8            | 0.15            |
| HEMOGLOBIN_MAX_VAL  | 12.81159644 | 12.79388359| -0.1382564381      | 72           | 149          | 221          | 4.27            |
| INR_MAX_VAL         | 3.376033846 | 3.084955418| -8.621904888       | 0            | 127          | 127          | 2.45            |
| LACTATE_MAX_VAL     | 4.828688815 | 4.614455277| -4.436681386       | 0            | 261          | 261          | 5.04            |
| PLATELET_MAX_VAL    | 415.308852  | 407.2819526| -1.932754211       | 0            | 214          | 214          | 4.13            |
| POTASSIUM_MAX_VAL   | 5.617823734 | 5.566002365| -0.9224456271      | 2            | 286          | 288          | 5.56            |
| SODIUM_MAX_VAL      | 145.7106128 | 145.5864484| -0.08521303837     | 86           | 143          | 229          | 4.42            |
| ANIONGAP_MIN_VAL    | 9.406805878 | 9.259109735| -1.570098759       | 34           | 170          | 204          | 3.94            |
| BUN_MIN_VAL         | 15.53362969 | 14.53718738| -6.414742227       | 0            | 239          | 239          | 4.62            |
| CHLORIDE_MIN_VAL    | 96.08157742 | 96.10867127| 0.02819879682      | 148          | 99           | 247          | 4.77            |
| CREATININE_MIN_VAL  | 0.9563780441| 0.8981654865| -6.086772683      | 0            | 261          | 261          | 5.04            |
| HEMOGLOBIN_MIN_VAL  | 8.044820255 | 8.014707788| -0.3743087626      | 69           | 186          | 255          | 4.93            |
| INR_MIN_VAL         | 1.158842638 | 1.138437361| -1.76083237        | 0            | 185          | 185          | 3.57            |
| LACTATE_MIN_VAL     | 1.308814626 | 1.221710625| -6.655182388       | 0            | 161          | 161          | 3.11            |
| PLATELET_MIN_VAL    | 127.1265945 | 123.6234585| -2.755628007       | 0            | 189          | 189          | 3.65            |
| POTASSIUM_MIN_VAL   | 3.196617704 | 3.179681691| -0.5298104002      | 67           | 168          | 235          | 4.54            |
| SODIUM_MIN_VAL      | 131.402861  | 131.5164675| 0.08645664798      | 208          | 63           | 271          | 5.24            |


## Exploratory Data Analysis

This section includes the exploratory data analysis

### Patient Cohort

![Patient Cohort Image](Report%20Figures/Patient%20Selection%20Figure.png)

### Feature Statistics

Below is a table of the numerical baseline characteristics, vital signs, laboratory parameters and statistic results of mimic-III patients with sepsis. The table highlights several significant differences in baseline characteristics, vital signs, and laboratory parameters between sepsis patients who survived and those who did not within 30 days. Key predictors of mortality include renal and metabolic dysfunction indicators (e.g., elevated BUN, lactate, and anion gap), hemodynamic instability (e.g., lower SBP and MAP), respiratory distress (e.g., lower oxygen saturation), coagulation abnormalities (e.g., elevated INR), older age, and shorter hospital stays.  

| Feature                             | Mean (`MORTALITY=0`) [95% CI]                      | Mean (`MORTALITY=1`) [95% CI]                      | p-value            |
|-------------------------------------|----------------------------------------|----------------------------------------|--------------------|
| BUN_MIN_VAL                         | 11.8391 (11.5640, 12.1142)             | 20.5947 (19.9426, 21.2469)             | 2.57E-115          |
| LACTATE_MAX_VAL                     | 3.9765 (3.8948, 4.0583)                | 6.0467 (5.8692, 6.2242)                | 5.23E-88           |
| SBP_MEAN                            | 116.4027 (115.9590, 116.8464)          | 107.2450 (106.4869, 108.0031)          | 1.43E-86           |
| LACTATE_MIN_VAL                     | 1.0718 (1.0557, 1.0879)                | 1.5583 (1.5135, 1.6032)                | 1.85E-81           |
| PLATELET_MAX_VAL                    | 441.4380 (434.7949, 448.0812)          | 330.5968 (321.2474, 339.9461)          | 4.63E-76           |
| DBP_MEAN                            | 60.3454 (60.0613, 60.6295)             | 55.5000 (55.0728, 55.9272)             | 1.16E-72           |
| MAP_MEAN                            | 76.7521 (76.4814, 77.0228)             | 72.0117 (71.5865, 72.4370)             | 5.97E-72           |
| ANIONGAP_MIN_VAL                    | 8.7372 (8.6631, 8.8112)                 | 10.4310 (10.2674, 10.5945)             | 2.59E-71           |
| INR_MIN_VAL                         | 1.0987 (1.0928, 1.1046)                 | 1.2277 (1.2137, 1.2416)                 | 3.47E-59           |
| ANIONGAP_MAX_VAL                    | 20.4475 (20.2887, 20.6063)             | 23.0914 (22.8143, 23.3685)             | 1.27E-56           |
| OXYGEN_SAT_MEAN                     | 97.0256 (96.9806, 97.0706)             | 95.9278 (95.8015, 96.0541)             | 1.07E-54           |
| BUN_MAX_VAL                         | 55.8206 (54.6891, 56.9521)             | 71.1694 (69.4175, 72.9213)             | 1.08E-45           |
| CREATININE_MIN_VAL                  | 0.8126 (0.7950, 0.8302)                 | 1.0903 (1.0554, 1.1252)                 | 1.63E-42           |
| POTASSIUM_MIN_VAL                   | 3.1267 (3.1137, 3.1396)                 | 3.2987 (3.2745, 3.3229)                 | 1.11E-33           |
| PLATELET_MIN_VAL                    | 131.1287 (128.6262, 133.6312)           | 106.7732 (102.7303, 110.8161)           | 2.32E-23           |
| TEMP_MEAN_C                         | 36.8956 (36.8784, 36.9128)              | 36.6986 (36.6639, 36.7332)              | 4.74E-23           |
| HEMOGLOBIN_MAX_VAL                  | 12.9470 (12.8892, 13.0047)              | 12.4502 (12.3625, 12.5379)              | 3.06E-20           |
| HEARTRATE_MEAN                      | 87.7993 (87.4007, 88.1978)              | 91.4300 (90.7714, 92.0887)              | 4.34E-20           |
| AGE_AT_ADMISSION                    | 66.3109 (65.7908, 66.8310)              | 69.6846 (68.9492, 70.4200)              | 2.58E-13           |
| RR_MEAN                             | 20.1052 (19.9986, 20.2117)              | 20.7528 (20.5759, 20.9297)              | 8.85E-10           |
| CREATININE_MAX_VAL                  | 3.0329 (2.9365, 3.1292)                  | 3.5043 (3.3746, 3.6340)                  | 1.14E-08           |
| INR_MAX_VAL                         | 2.9458 (2.8593, 3.0322)                  | 3.3974 (3.2649, 3.5299)                  | 2.35E-08           |
| LOS_ICU_MEAN                        | 5.7327 (5.5569, 5.9084)                  | 6.7263 (6.4204, 7.0322)                  | 3.63E-08           |
| CHLORIDE_MAX_VAL                    | 113.0423 (112.8532, 113.2313)            | 112.0484 (111.7189, 112.3778)            | 3.07E-07           |
| LOS                                 | 14.0054 (13.6193, 14.3915)               | 12.2591 (11.6275, 12.8906)               | 3.88E-06           |
| TEMP_MIN_C                          | 34.6777 (34.5590, 34.7963)               | 34.1539 (33.9566, 34.3511)               | 8.39E-06           |
| SODIUM_MAX_VAL                      | 145.8018 (145.6469, 145.9568)            | 145.1029 (144.8254, 145.3804)            | 1.67E-05           |
| POTASSIUM_MAX_VAL                   | 5.5335 (5.4959, 5.5711)                  | 5.6390 (5.5861, 5.6920)                  | 0.001451080247     |
| DIABETES                            | 0.3809 (0.3650, 0.3968)                  | 0.3561 (0.3326, 0.3796)                  | 0.08707751955      |
| HEMOGLOBIN_MIN_VAL                  | 7.9936 (7.9412, 8.0459)                  | 8.0622 (7.9807, 8.1437)                  | 0.1646019274       |
| SODIUM_MIN_VAL                      | 131.5808 (131.4097, 131.7518)            | 131.3721 (131.0853, 131.6589)            | 0.2204434812       |
| TEMP_MAX_C                          | 39.6228 (39.3885, 39.8571)               | 39.3669 (39.0131, 39.7207)               | 0.2370434699       |
| WEIGHT_MEAN                         | 81.9442 (81.2536, 82.6348)               | 81.9198 (80.8720, 82.9676)               | 0.9695997544       |
| CHLORIDE_MIN_VAL                    | 96.1065 (95.9060, 96.3070)               | 96.1136 (95.7766, 96.4507)               | 0.9715199933       |


## Model Development and Evaluation


## Discussion

## Conclusion

## References
- Hou, N., Li, M., He, L. et al. Predicting 30-days mortality for MIMIC-III patients with sepsis-3: a machine learning approach using XGboost. J Transl Med 18, 462 (2020). https://doi.org/10.1186/s12967-020-02620-5
- https://cloud.google.com/bigquery/docs/python-libraries
- https://xgboost.readthedocs.io/en/stable/
- Johnson, A., Pollard, T., Shen, L. et al. MIMIC-III, a freely accessible critical care database. Sci Data 3, 160035 (2016). https://doi.org/10.1038/sdata.2016.35
- https://www.who.int/news-room/fact-sheets/detail/sepsis
- Duncan, C. F., Youngstein, T., Kirrane, M. D., & Lonsdale, D. O. (2021). Diagnostic Challenges in Sepsis. Current infectious disease reports, 23(12), 22. https://doi.org/10.1007/s11908-021-00765-y
- R.H. Riffenburgh, Chapter 28 - Methods You Might Meet, But Not Every Day, Editor(s): R.H. Riffenburgh, Statistics in Medicine (Third Edition), Academic Press, 2012, Pages 581-591, ISBN 9780123848642, https://doi.org/10.1016/B978-0-12-384864-2.00028-7. (https://www.sciencedirect.com/science/article/pii/B9780123848642000287)

