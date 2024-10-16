# 650 Project - Predicting 30‑days mortality for MIMIC‑III patients with sepsis‑3

**Authors**: Joshua Cabal, Viridiana Radillo, Ida Karima
California State University, Northridge
December 4, 2024

---

Table of Contents:
- [650 Project - Predicting 30‑days mortality for MIMIC‑III patients with sepsis‑3](#650-project---predicting-30days-mortality-for-mimiciii-patients-with-sepsis3)
  - [Executive Sumamry](#executive-sumamry)
  - [Introduction](#introduction)
    - [Background](#background)
    - [Challenges in Predicting Sepsis Outcomes](#challenges-in-predicting-sepsis-outcomes)
    - [Advancements in Machine Learning](#advancements-in-machine-learning)
    - [Purpose of the Study](#purpose-of-the-study)
    - [Objectives](#objectives)
    - [Significance](#significance)
  - [Methodology](#methodology)
  - [Data Description](#data-description)
    - [Patient Selection Criteria](#patient-selection-criteria)
      - [Queries](#queries)
    - [Data Extraction](#data-extraction)
    - [Data Aggregation](#data-aggregation)
  - [Analysis and Findings](#analysis-and-findings)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Discussion](#discussion)
  - [Recommendations](#recommendations)
  - [Conclusion](#conclusion)
  - [Appendices](#appendices)
  - [References](#references)

---

## Executive Sumamry

Sepsis is a critical global health issue with high mortality rates, particularly among ICU patients. Early prediction of sepsis outcomes is essential for improving patient care and survival rates. This report replicates the study by Hou et al. (2020), aiming to **develop a predictive model for 30-day mortality in sepsis-3 patients** using the MIMIC-III database.

Using machine learning techniques, specifically the XGBoost algorithm, we constructed a predictive model and compared its performance with traditional logistic regression and SAPS-II scoring models. Our findings indicate that the XGBoost model outperforms the conventional models, demonstrating higher accuracy and better predictive capabilities.

This study reinforces the potential of machine learning approaches in clinical settings, suggesting that the XGBoost model could assist clinicians in making informed decisions and tailoring precise treatments for sepsis patients.

## Introduction

### Background

Sepsis, a life-threatening organ dysfunction caused by a dysregulated host response to infection, remains a leading cause of mortality in intensive care units (ICUs) worldwide. With over 5 million deaths annually, the burden of sepsis on healthcare systems is profound, necessitating improved strategies for early detection and management.

### Challenges in Predicting Sepsis Outcomes

Traditional prognostic tools, such as serum biomarkers and scoring systems like APACHE-II and SAPS-II, have limitations in sensitivity, specificity, and adaptability. These methods often rely on linear assumptions and may not fully capture the complex interactions of clinical variables in sepsis patients.

### Advancements in Machine Learning

Machine learning techniques offer flexible and powerful alternatives for predictive modeling in healthcare. Algorithms like XGBoost have shown promise in handling large datasets and identifying nonlinear relationships among variables, potentially improving predictive accuracy in clinical outcomes.

### Purpose of the Study

This report aims to replicate the study conducted by Hou et al. (2020), which developed an XGBoost-based machine learning model to predict 30-day mortality in sepsis-3 patients using data from the MIMIC-III database. By replicating this study, we seek to validate their findings and assess the model's applicability in predicting sepsis outcomes.

### Objectives

- To develop a machine learning model using XGBoost for predicting 30-day mortality in sepsis-3 patients.
- To compare the performance of the XGBoost model with traditional logistic regression and SAPS-II score models.
- To evaluate the clinical utility of the XGBoost model through validation techniques.

### Significance

Validating and potentially enhancing predictive models for sepsis outcomes can aid clinicians in early identification of high-risk patients, enabling timely interventions and personalized treatment strategies. This study contributes to the growing body of evidence supporting the integration of machine learning in critical care settings.

## Methodology

## Data Description

### Patient Selection Criteria

The following criteria were used in the selection of the patient records:

- Patient must be diagnosed with sepsis. ICD9 Codes:
    - 99591, Sepsis
    - 99592, Severe sepsis
- Patient must be at least 18 years old
- Patient must have demographic data
- Patient must have related lab test results
- Patient must have no more than 20% of data missing

#### Queries
- Basic Patient Query:
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

### Data Extraction

We started with extracting all of the data related to patients diagnosed with sepsis:

- Chart Events Query:

```SQL
SELECT *
FROM `physionet-data.mimiciii_clinical.chartevents` AS C
INNER JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` AS D ON D.SUBJECT_ID = C.SUBJECT_ID
WHERE D.ICD9_CODE IN ('99591', '99592')
```

- Heart Rate Average Query:
```SQL
SELECT C.SUBJECT_ID,ITEM.LABEL, AVG(C.VALUENUM) AS HEARTRATE
FROM `physionet-data.mimiciii_clinical.chartevents` AS C
INNER JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` AS D ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN `physionet-data.mimiciii_clinical.d_items` AS ITEM ON ITEM.ITEMID = C.ITEMID
WHERE D.ICD9_CODE IN ('99591', '99592') AND (C.ITEMID  IN (211, 220045))
GROUP BY C.SUBJECT_ID, ITEM.LABEL
ORDER BY C.SUBJECT_ID ASC
```

### Data Aggregation

Because many of the features are captured periodically, such as vital signs and lab events, these features are incorporated using their minimum, maximum, and mean.

## Analysis and Findings

### Exploratory Data Analysis

## Discussion

## Recommendations

## Conclusion

## Appendices

## References
- https://cloud.google.com/bigquery/docs/python-libraries
