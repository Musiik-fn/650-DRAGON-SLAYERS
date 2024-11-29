# 650 Project - Predicting 30‑days mortality for MIMIC‑III patients with sepsis‑3

**Authors**: Joshua Cabal,  Ida Karima, Viridiana Radillo,

California State University, Northridge

December 4, 2024

---

Table of Contents:
- [650 Project - Predicting 30‑days mortality for MIMIC‑III patients with sepsis‑3](#650-project---predicting-30days-mortality-for-mimiciii-patients-with-sepsis3)
  - [Abstract](#abstract)
  - [Introduction](#introduction)
    - [Background](#background)
    - [Challenges in Predicting Sepsis Outcomes](#challenges-in-predicting-sepsis-outcomes)
    - [Purpose of the Study](#purpose-of-the-study)
    - [Objectives](#objectives)
    - [Significance](#significance)
  - [Methodology](#methodology)
    - [Database Interfacing](#database-interfacing)
    - [Data Extraction](#data-extraction)
    - [Patient Selection Criteria and Query](#patient-selection-criteria-and-query)
    - [Feature Selection](#feature-selection)
    - [Data Aggregation](#data-aggregation)
    - [Data Preprocessing](#data-preprocessing)
      - [Defining Outcome Variable: 30-Day Mortality](#defining-outcome-variable-30-day-mortality)
      - [Categorical Feature Encoding](#categorical-feature-encoding)
      - [Handling `NULL` Values](#handling-null-values)
      - [Handling Outliers](#handling-outliers)
  - [Results](#results)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
      - [Patient Cohort](#patient-cohort)
      - [Feature Statistics](#feature-statistics)
    - [Predictive Models](#predictive-models)
      - [Logistic Regression](#logistic-regression)
      - [Random Forest](#random-forest)
      - [XGBoost](#xgboost)
    - [Model Comparison](#model-comparison)
      - [Selected Features](#selected-features)
  - [Conclusion](#conclusion)
    - [Replication of Selected Paper](#replication-of-selected-paper)
    - [Key Findings and Healthcare Implications](#key-findings-and-healthcare-implications)
  - [Lessons Learned](#lessons-learned)
  - [Appendix](#appendix)
  - [References](#references)

---

## Abstract

**Background**: Sepsis is a significant cause of in-hospital mortality, particularly among ICU patients. Early prediction of sepsis is essential, as prompt and appropriate treatment can improve survival outcomes. Machine learning methods are flexible prediction algorithms with potential advantages over conventional regression and scoring systems. The aim of this study were to replicate and validate the findings performed by [Hou et al. (2020)](https://doi.org/10.1186/s12967-020-02620-5), which found that XGBoost performed better than traditional predictive models.

**Methods**: Using the MIMIC-III v1.4 database, we identified patients with sepsis-3. The data were split into two groups based on death or survival within 30 days. Variables selected based on clinical significance and availability through stepwise analysis were compared between groups. Three predictive models were constructed using R software: a conventional logistic regression model, the SAPS-II score prediction model, and an XGBoost algorithm model. The performances of the three models were tested and compared using the area under the receiver operating characteristic curve (AUC) and average precision through precision-recall curves. Finally, a clinical impact curve were used to validate the model.

**Results**: A total of 4,555 sepsis-3 patients were included in the study, among whom 1,274 patients died and 3,281 survived within 30 days. According to the results of the AUCs—0.7513 for the logistic regression model, 0.7561 for the random forest model, and 0.7832 for the XGBoost model—and average precision, the XGBoost model performed best. The clinical impact curve verified that the XGBoost model possesses predictive value.

**Conclusions**: A more significant predictive model can be built using the machine learning technique XGBoost. This XGBoost model may prove clinically useful and assist clinicians in tailoring precise management and therapy for patients with sepsis-3.

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

The primary code used in this project can be found in the directory [FILES > Code](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/tree/main/FILES/Code).

### Database Interfacing

Similar to the source paper, we used the Medical Information Mart for Intensive Care III database version 1.4 ([MIMIC III v1.4](https://doi.org/10.1038/sdata.2016.35)) for the study. MIMIC-III, a publicly available single-center critical care database which was approved by the Institutional Review Boards of Beth Israel Deaconess Medical Center (BIDMC, Boston, MA, USA) and the Massachusetts Institute of Technology (MIT, Cambridge, MA, USA), includes information on 46,520 patients who were admitted to various ICUs of BIDMC in Boston, Massachusetts from 2001 to 2012. The data was accessed via the Google BigQuery cloud platform and subsequently extracted and processed using SQL and Python. Additionally, the models were developed and evaluated in Python. This approach contrasts with the target paper, which primarily utilized R for these tasks.


### Data Extraction

We first extracted the given admission and patient data of all patients who were diagnosed with Sepsis. After this, we extracted the related baseline, vital, and laboratory data related to each sepsis patient. All variables were loaded into their own dataframe and eventually all dataframes were merged onto the patient list which was initially extracted. After this, the data transformation steps included handling inconsistencies, missing values, outliers, and categorical feature encoding. Finally, the 30 day mortality column is defined.

![Patient Cohort Image](Report%20Figures/Data%20Preprocessing%20Figure(1).png)

### Patient Selection Criteria and Query

The following criteria were used in the selection of the patient records:

- Patient must be diagnosed with sepsis. ICD9 Codes:
    - 99591, Sepsis
    - 99592, Severe sepsis
- Patient must be at least 18 years old
- Patient must have demographic data
- Patient must have related lab test results
- Patient must have no more than 20% of data missing

The baseline patient query can be found in the file [Data_Extraction.sql](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/Data_Extraction.sql). The criteria mentioned above were applied after the dataframe merge was completed.


### Feature Selection

The features were divided into the following categories: **baseline variables, vital signs, and laboratory parameters**. The variables were chosen directly from the paper which selected the features based on clinical significance and availability by stepwise analysis between the groups. For the variables that were captured periodically, such as heart rate, they were aggregated and incorporated into the model using **min**, **mean**, and **maximum values**. 


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

Because many of the features are captured periodically, such as vital signs and lab events, these features are incorporated using their minimum, maximum, and mean. A query was written to extract each feature listed in the feature table, and then the results were merged on the sepsis patient table on the `SUBJECT_ID`. All SQL statements can be found in the file [Data_Extraction.sql](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/Data_Extraction.sql). 

### Data Preprocessing

#### Defining Outcome Variable: 30-Day Mortality
The sepsis patients were divided into two groups based on their 30 day mortality:
1. `MORTALITY = 0`: Died within 30 days
2. `MORTALITY = 1`: Survived within 30 days


The statistics related to this will be provided in the exploratory data analysis section of this paper. 


#### Categorical Feature Encoding

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

After the ethnicity grouping was completed, the following results were observed:

| Ethnicity                                       | Count |
|-------------------------------------------------|-------|
| WHITE                                           | 3,360 |
| BLACK OR AFRICAN AMERICAN                       | 436   |
| UNKNOWN/NOT SPECIFIED/DECLINED                  | 340   |
| ASIAN                                           | 148   |
| HISPANIC OR LATINO                              | 144   |
| OTHER                                           | 117   |
| AMERICAN INDIAN OR ALASKA NATIVE                 | 4     |
| MIDDLE EASTERN                                  | 4     |
| NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER       | 2     |


#### Handling `NULL` Values
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

#### Handling Outliers

The outliers were handled using [winsorization](https://www.sciencedirect.com/science/article/abs/pii/B9780123848642000287).  Through this method, extreme values were limited to a chosen distance of 3 standard deviations from the mean. The function which handled this is, `preprocess_outliers()`, defined in the file [dragonFunctions.py](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/dragonFunctions.py). The function also tracks statsitcs of the variables before and after the operation. The results are contained in the file named [Outlier_Report.csv](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Data/Outlier_Report.csv).


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

## Results

### Exploratory Data Analysis


#### Patient Cohort

After all filtering, our final cohort contains 1274 patients who died within 30 days and 3281 patients who survived within 30 days. 

![Patient Cohort Image](Report%20Figures/Patient%20Selection%20Figure.png)

#### Feature Statistics

Below is a table of the numerical baseline characteristics, vital signs, laboratory parameters and statistic results of mimic-III patients with sepsis. The table highlights several significant differences in baseline characteristics, vital signs, and laboratory parameters between sepsis patients who survived and those who did not within 30 days. Key predictors of mortality include older age, longer hospital and ICU stays, higher systolic and mean arterial pressures, lower heart rates, elevated BUN, lactate, anion gap, INR, and creatinine levels, as well as significant coagulation abnormalities indicated by platelet counts.

| Column Index  | Feature                                                 | Mean (Death within 30 days) [95% CI]                         | Mean (Survival within 30 days) [95% CI]                         | p-value                  |
|----|---------------------------------------------------------|-------------------------------------------|-------------------------------------------|--------------------------|
| 18 | ANIONGAP_MIN_VAL                                        | 8.5226 (8.4010, 8.6442)                    | 9.8713 (9.7721, 9.9706)                    | 5.804219462456842e-61    |
| 1  | LOS                                                     | 18.5120 (17.7062, 19.3177)                | 11.3806 (11.0295, 11.7316)                | 1.9994524351984268e-53    |
| 24 | LACTATE_MIN_VAL                                         | 1.0757 (1.0487, 1.1027)                    | 1.3592 (1.3321, 1.3863)                    | 1.1754059843635527e-46   |
| 26 | POTASSIUM_MIN_VAL                                       | 3.0785 (3.0576, 3.0995)                    | 3.2703 (3.2551, 3.2856)                    | 3.694125814703512e-46    |
| 11 | OXYGEN_SAT_MEAN                                         | 97.1715 (97.0979, 97.2451)                | 96.3800 (96.3059, 96.4541)                | 1.3911908285127357e-48   |
| 17 | SODIUM_MAX_VAL                                          | 146.6111 (146.3550, 146.8673)             | 144.7146 (144.5420, 144.8873)             | 1.653643639974896e-32    |
| 22 | HEMOGLOBIN_MIN_VAL                                      | 7.7102 (7.6326, 7.7879)                    | 8.3235 (8.2666, 8.3803)                    | 7.003674251346475e-35    |
| 23 | INR_MIN_VAL                                             | 1.0944 (1.0845, 1.1044)                    | 1.1784 (1.1699, 1.1869)                    | 2.513150150344204e-35    |
| 19 | BUN_MIN_VAL                                             | 13.0178 (12.5203, 13.5152)                | 16.2705 (15.8542, 16.6867)                | 1.6903585610211654e-22   |
| 27 | SODIUM_MIN_VAL                                          | 130.9360 (130.6529, 131.2192)              | 132.2608 (132.0762, 132.4454)              | 2.1554673074035844e-14   |
| 0  | AGE_AT_ADMISSION                                       | 70.2793 (69.4675, 71.0911)                | 66.3547 (65.8040, 66.9055)                | 6.2423475069410055e-15   |
| 20 | CHLORIDE_MIN_VAL                                        | 95.4482 (95.1205, 95.7758)                 | 96.9124 (96.6922, 97.1326)                 | 4.640940223674255e-13    |
| 4  | HEARTRATE_MEAN                                          | 87.0909 (86.4337, 87.7481)                | 89.9568 (89.5002, 90.4135)                | 2.743262197438905e-12    |
| 2  | LOS_ICU_MEAN                                            | 7.1504 (6.7878, 7.5131)                    | 5.6759 (5.4888, 5.8630)                    | 1.8824470974372263e-12   |
| 5  | SBP_MEAN                                                | 115.8958 (115.1019, 116.6897)             | 112.6069 (112.0836, 113.1301)             | 1.4512206581707315e-11   |
| 21 | CREATININE_MIN_VAL                                      | 0.8351 (0.8033, 0.8668)                    | 0.9469 (0.9253, 0.9684)                    | 1.2462387708005441e-08   |
| 14 | BUN_MAX_VAL                                             | 63.7555 (61.8355, 65.6755)                | 56.9016 (55.7133, 58.0899)                | 3.0159367740247727e-09   |
| 16 | INR_MAX_VAL                                             | 3.2497 (3.0991, 3.4004)                    | 2.8127 (2.7315, 2.8939)                    | 5.972261612669238e-07    |
| 25 | PLATELET_MIN_VAL                                        | 117.7276 (113.4049, 122.0504)              | 130.2693 (127.4697, 133.0690)              | 1.8898490558943614e-06   |
| 3  | WEIGHT_MEAN                                             | 79.9468 (78.8430, 81.0506)                | 82.9039 (82.1627, 83.6451)                | 1.337480960599974e-05    |
| 9  | TEMP_MIN_C                                              | 34.3334 (34.1245, 34.5424)                | 34.7919 (34.6779, 34.9059)                | 0.00016306792291858775   |
| 7  | MAP_MEAN                                                | 76.0140 (75.5646, 76.4634)                | 75.0111 (74.6983, 75.3239)                | 0.0003334220111951916    |
| 8  | RR_MEAN                                                 | 20.1908 (20.0177, 20.3639)                | 20.4973 (20.3766, 20.6181)                | 0.004415739666238388     |
| 12 | DIABETES                                                | 0.3885 (0.3617, 0.4153)                    | 0.3472 (0.3309, 0.3634)                    | 0.009707880047703427     |
| 15 | HEMOGLOBIN_MAX_VAL                                      | 12.7976 (12.7051, 12.8902)                | 12.6557 (12.5938, 12.7176)                | 0.012476564718123813     |
| 6  | DBP_MEAN                                                | 58.5420 (58.0619, 59.0220)                | 58.9291 (58.6115, 59.2467)                | 0.1872252166711786       |
| 10 | TEMP_MAX_C                                              | 39.3719 (39.0492, 39.6947)                | 39.1706 (38.9667, 39.3745)                | 0.3011192382325408       |
| 13 | ANIONGAP_MAX_VAL                                        | 21.0742 (20.8063, 21.3422)                | 20.9933 (20.8107, 21.1759)                | 0.6244584585997122       |

Below are the distributions of the features with the lowest p-values.

![EDA](Report%20Figures/EDA%20Visuals/AGE.png)
![EDA](Report%20Figures/EDA%20Visuals/ANIONGAP_MIN.png)
![EDA](Report%20Figures/EDA%20Visuals/LACTATE_MIN.png)
![EDA](Report%20Figures/EDA%20Visuals/LOS.png)
![EDA](Report%20Figures/EDA%20Visuals/O2SAT.png)
![EDA](Report%20Figures/EDA%20Visuals/POTASSIUM_MIN.png)
![EDA](Report%20Figures/EDA%20Visuals/SODIUM_MIN.png)

All histograms and box plots can be found in [FILES > Code > 650EDA.ipynb](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/650EDA.ipynb)

For the categorical features, a chi-squared test was applied and the following results were given: 
| Variable  | Chi-squared Statistic     | P-value                | Degrees of Freedom |
|-----------|---------------------------|------------------------|--------------------|
| ETHNICITY | 21.940778878649233        | 0.005026763602206442   | 8                  |
| GENDER    | 1.4287106646326184        | 0.23197497397598826    | 1                  |
| ADMISSION_TYPE | 8.743278584659594       | 0.012630518475803666    | 2                  |

The chi-squared analysis reveals that ethnicity and admission type are significantly associated with 30-day mortality in sepsis patients, while gender is not.


### Predictive Models
The data was standardized using `StandardScaler()`. The coefficient/feature importance values for each model are saved in the `Data` folder. The code related to model development can be found in the file [650Models.ipynb](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/650Models.ipynb). This notebook utilizes many custom functions defined in the library file [dragonFunctions.py](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/dragonFunctions.py).

#### Logistic Regression


![LR_ROC](Report%20Figures/Plots/ROC_Curve_Logistic%20Regression.png)

| Feature                                             | Coefficient         | Abs Coefficient     |
|-----------------------------------------------------|---------------------|---------------------|
| LOS                                                 | -0.4932606042579501 | 0.4932606042579501  |
| ETHNICITY_CONSOLIDATED_OTHER                        | 0.4496618910098512  | 0.4496618910098512  |
| ETHNICITY_CONSOLIDATED_WHITE                        | -0.40819109301538914| 0.40819109301538914 |
| AGE_AT_ADMISSION                                    | -0.3969062460840332 | 0.3969062460840332  |
| ADMISSION_TYPE_URGENT                               | 0.3341514858193994  | 0.3341514858193994  |
| OXYGEN_SAT_MEAN                                     | -0.2813922984121403 | 0.2813922984121403  |
| GENDER_M                                            | -0.27312351145168834| 0.27312351145168834 |
| BUN_MIN_VAL                                         | 0.24677433273110874 | 0.24677433273110874 |
| ETHNICITY_CONSOLIDATED_BLACK OR AFRICAN AMERICAN    | -0.24360627899130383| 0.24360627899130383 |
| POTASSIUM_MIN_VAL                                   | 0.22354275485744332 | 0.22354275485744332 |
| ETHNICITY_CONSOLIDATED_HISPANIC OR LATINO           | 0.16890327194533392 | 0.16890327194533392 |
| LOS_ICU_MEAN                                        | 0.1576491889366517  | 0.1576491889366517  |
| CREATININE_MIN_VAL                                  | -0.1404407470985917 | 0.1404407470985917  |
| WEIGHT_MEAN                                         | 0.139056926608819   | 0.139056926608819   |
| INR_MIN_VAL                                         | 0.1371899462909558  | 0.1371899462909558  |

![LR_Coeff](Report%20Figures/Plots/Feature_Coefficients_Logistic%20Regression.png)

![CM_LR](Report%20Figures/Plots/Confusion_Matrix_Logistic%20Regression_Cross_Validated_Training.png)

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.64      | 0.29   | 0.40     | 255     |
| 1             | 0.77      | 0.93   | 0.85     | 656     |
| **Accuracy**  |           |        | 0.76     | 911     |
| **Macro Avg** | 0.70      | 0.61   | 0.62     | 911     |
| **Weighted Avg** | 0.73   | 0.76   | 0.72     | 911     |


#### Random Forest
![RF_ROC](Report%20Figures/Plots/ROC_Curve_Random%20Forest.png)
Feature Importance Table:
| Feature                                             | Importance          |
|-----------------------------------------------------|---------------------|
| LOS                                                 | 0.0751990622383627  |
| OXYGEN_SAT_MEAN                                     | 0.05343446104510928 |
| AGE_AT_ADMISSION                                    | 0.04336838087830143 |
| SBP_MEAN                                            | 0.041516251625344364|
| PLATELET_MIN_VAL                                    | 0.03997843273489437 |
| HEARTRATE_MEAN                                      | 0.0395753096345336  |
| TEMP_MIN_C                                          | 0.03936314414109013 |
| WEIGHT_MEAN                                         | 0.03915667520754803 |
| DBP_MEAN                                            | 0.03862787206256848 |
| LOS_ICU_MEAN                                        | 0.03814046174282419 |
| RR_MEAN                                             | 0.037662277725471434|
| HEMOGLOBIN_MIN_VAL                                  | 0.035895643404254864|
| TEMP_MAX_C                                          | 0.03564243164480918 |
| BUN_MAX_VAL                                         | 0.035147285383892744|
| MAP_MEAN                                            | 0.034835832439807125|

![RF_ROC](Report%20Figures/Plots/Feature_Importances_Random%20Forest.png)
![RF_ROC](Report%20Figures/Plots/Confusion_Matrix_Random%20Forest_Cross_Validated_Training.png)

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0             | 0.72      | 0.28   | 0.40     | 255     |
| 1             | 0.77      | 0.96   | 0.86     | 656     |
| **Accuracy**  |           |        | 0.77     | 911     |
| **Macro Avg** | 0.75      | 0.62   | 0.63     | 911     |
| **Weighted Avg** | 0.76   | 0.77   | 0.73     | 911     |

#### XGBoost

From the [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/),  XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.

Through `GridSearchCV()`, the best following XGBoost parameters were chosen: 
```
Best Parameters: {'colsample_bytree': 0.6, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}
```

![XGBoost ROC](Report%20Figures/Plots/ROC_Curve_XGBoost.png)

Feature Gain Table:
| Feature                                             | Gain                | Normalized Gain     |
|-----------------------------------------------------|---------------------|---------------------|
| LOS                                                 | 11.8303804397583    | 0.07137593          |
| ANIONGAP_MIN_VAL                                    | 8.511284828186035   | 0.05135092          |
| AGE_AT_ADMISSION                                    | 6.546313762664795   | 0.03949571          |
| OXYGEN_SAT_MEAN                                     | 6.159742832183838   | 0.03716342          |
| ETHNICITY_CONSOLIDATED_ASIAN                        | 5.747664451599121   | 0.03467724          |
| POTASSIUM_MIN_VAL                                   | 5.5271782875061035  | 0.03334698          |
| INR_MIN_VAL                                         | 5.065011978149414   | 0.03055861          |
| ETHNICITY_CONSOLIDATED_WHITE                        | 4.973119258880615   | 0.03000419          |
| TEMP_MIN_C                                          | 4.901345729827881   | 0.02957116          |
| SODIUM_MIN_VAL                                      | 4.747461318969727   | 0.02864274          |
| ETHNICITY_CONSOLIDATED_HISPANIC OR LATINO           | 4.652392387390137   | 0.02806916          |
| BUN_MIN_VAL                                         | 4.564362049102783   | 0.02753805          |
| BUN_MAX_VAL                                         | 4.491528511047363   | 0.02709862          |
| LACTATE_MIN_VAL                                     | 4.418938636779785   | 0.02666067          |
| MAP_MEAN                                            | 4.241907596588135   | 0.02559259          |
| CREATININE_MIN_VAL                                  | 4.1905012130737305  | 0.02528244          |

![XGBoost Feature Gain](Report%20Figures/Plots/Feature_Importances_XGBoost.png)

![XGBoost Confusion Matrix](Report%20Figures/Plots/Confusion_Matrix_XGBoost_Cross_Validated_Training.png)

### Model Comparison

Plotting the Average Precision and Receiver Operating Characteristic curves for all models onto a single graph, we have the following results:

![AP Curve Comparison](Report%20Figures/Plots/Combined_ROC_Curves.png)

![ROC Curve Comparison](Report%20Figures/Plots/Combined_Precision_Recall_Curves.png)

XGBoost attains the highest values for both AUC and AP, and is therefore the most optimal model given the dataset. 

![CIC_XGB](Report%20Figures/Plots/CIC_XGB.png)
Clinical impact curve (CIC) of XGboost model. The red curve (number of high-risk individuals) indicates the number of people who are classified as positive (high risk) by the model at each threshold probability; the blue curve (number of high-risk individuals with outcome) is the number of true positives at each threshold probability. The Clinical Impact Curve demonstrates that the model has clinical utility by maintaining a significant gap between the true positives (blue curve) and the high-risk cases (red curve), indicating effective identification of true positives while minimizing false positives. The blue curve remains high across clinically relevant threshold probabilities (0.2–0.6), capturing a substantial number of true positive cases, which is critical in medical applications. Additionally, the narrow confidence intervals for both curves suggest the model’s predictions are consistent and reliable, further supporting its practical applicability. Overall, the model effectively balances sensitivity and specificity, making it suitable for clinical decision-making.


#### Selected Features
Amongst all models, the baseline features of AGE and LOS (Length of Stay) were significant features. In terms of the vitals and laboratory signs, OXYGEN_SAT_MEAN stands out as the most important vital sign, consistently appearing as a top predictor. Other vital signs and lab values such as BUN levels, POTASSIUM_MIN_VAL, INR_MIN_VAL, CREATININE_MIN_VAL, TEMP_MIN_C, MAP_MEAN, and WEIGHT_MEAN are also significant in multiple models, reinforcing their relevance in predicting patient outcomes.


## Conclusion

In conclusion, the XGBoost model outperforms conventional logistic regression and random forest models. This replication confirms that the XGBoost model has the potential to be clinically beneficial, aiding healthcare professionals in providing precise management and treatment for patients with sepsis, which is crucial for enhancing the likelihood of patient survival.

### Replication of Selected Paper
Our developed models were unable to achieve the same Area Under the Curve (AUC) scores as those reported in the replicated paper. We have identified the key factors which may have contributed to this discrepancy:

- **Different Definitions of 30-Day Mortality**: We used Admission Time and Date of Death to calculuate the `MORTALITY` Boolean, but the paper may have used a different definition, such as using Length of Stay. The paper did not state how this was defined.
- **Outlier Handling Methods**: We used winsorization to handle outlier values, but the paper did not specify if or how these values were handled. 
- **Incomplete Feature Extraction**: Not all features utilized in the replicated paper were successfully extracted from the dataset.

Because the original paper did not provide the associated code, certain assumptions were made during replication. Crucially, the paper did not include a list of SQL queries or ITEM_IDs used for data extraction. Many ITEM_IDs correspond to a single feature, and without the exact ITEM_IDs used by the original study, our predictor variables were likely not the same.

For instance, when attempting to extract urine output data using LAB ITEMS 51108 and 51109, over 80% of the entries were NULL. Urine output was a significant feature in the original paper, but we could not determine the exact method they used to extract this data. It is likely that the relevant data resides under different ITEM_IDs, which we were unable to identify. Such challenges likely contributed to the differences in our results.

Despite these limitations, our XGBoost model still outperformed traditional models, aligning with the primary objective of the replicated paper.

### Key Findings and Healthcare Implications

Our replication demonstrated that the XGBoost machine learning model outperforms traditional Logistic Regression and Random Forest models in predicting 30-day mortality among sepsis-3 patients using the MIMIC-III database. Specifically, XGBoost achieved the highest Area Under the Receiver Operating Characteristic Curve (AUC) of 0.7832 and the strongest Average Precision (AP) score, indicating superior discrimination and precision-recall performance. Key predictors identified across models included Length of Stay (LOS), Age at Admission, Oxygen Saturation (OXYGEN_SAT_MEAN), and various biochemical markers such as Blood Urea Nitrogen (BUN) and Lactate levels. Additionally, the clinical utility of the model was verified with the clinical impact curve. 

The healthcare implications of these findings are significant. Implementing the XGBoost model in clinical settings can improve risk stratification, enabling healthcare providers to identify high-risk sepsis patients more accurately and allocate resources more effectively. This leads to better patient triage and timely interventions, potentially reducing mortality rates and enhancing patient outcomes. Moreover, the identification of critical predictors highlights areas for targeted clinical monitoring and intervention, addressing underlying factors that contribute to sepsis mortality. By integrating such advanced predictive models into Clinical Decision Support Systems (CDSS), hospitals can enhance decision-making processes, ensure equitable care across diverse patient populations, and ultimately contribute to more efficient and effective sepsis management protocols.


## Lessons Learned

As a team, we collectively leaned the following lessons:

- **Database Initialization**: We learned to initialize and work with databases both locally and on the cloud. We learned this through following the great MIMIC-III documentation and also learned about the trade offs between each choice. We believe working with the cloud instance was the best route given the project requirements. 
- **Secure Data Connections**: Established secure connections to cloud instances and imported data directly into notebooks. We adhered to best practices for securely transferring data, initializing application default credentials, and configuring our workflows to prevent exposure of sensitive information such as keys, secrets, or passwords.
- **Reproducibility in Research**: As a group, we emphasized the importance of verifying the findings of the selected research papers. By working together to replicate methodologies as accurately as possible, we highlighted the need for open-sourced code to improve transparency, visibility, and peer review in rigorous medical research.
- **Project Management and Accountability**: We developed effective strategies for managing tasks, assigning responsibilities, and ensuring accountability given our working schedules. Our collaboration strengthened our ability to track progress and meet project objectives efficiently.
- **Learning New Tools**: Each of us explored the evaluation metrics used in the target paper, such as average precision through precision-recall curves and clinical impact curves.

## Appendix

- [SQL Statements](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/Data_Extraction.sql)
- [Data Extraction Notebook](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/650DataExtract.ipynb)
- [Exploratory Data Analysis Notebook](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/650EDA.ipynb)
- [Models Notebook](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/650Models.ipynb)
- [Function Library](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/blob/main/FILES/Code/dragonFunctions.py)

## References
- Hou, N., Li, M., He, L. et al. Predicting 30-days mortality for MIMIC-III patients with sepsis-3: a machine learning approach using XGboost. J Transl Med 18, 462 (2020). https://doi.org/10.1186/s12967-020-02620-5
- https://cloud.google.com/bigquery/docs/python-libraries
- https://xgboost.readthedocs.io/en/stable/
- Johnson, A., Pollard, T., Shen, L. et al. MIMIC-III, a freely accessible critical care database. Sci Data 3, 160035 (2016). https://doi.org/10.1038/sdata.2016.35
- https://www.who.int/news-room/fact-sheets/detail/sepsis
- Duncan, C. F., Youngstein, T., Kirrane, M. D., & Lonsdale, D. O. (2021). Diagnostic Challenges in Sepsis. Current infectious disease reports, 23(12), 22. https://doi.org/10.1007/s11908-021-00765-y
- R.H. Riffenburgh, Chapter 28 - Methods You Might Meet, But Not Every Day, Editor(s): R.H. Riffenburgh, Statistics in Medicine (Third Edition), Academic Press, 2012, Pages 581-591, ISBN 9780123848642, https://doi.org/10.1016/B978-0-12-384864-2.00028-7. (https://www.sciencedirect.com/science/article/pii/B9780123848642000287)
