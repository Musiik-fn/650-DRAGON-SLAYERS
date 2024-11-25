-- ==============================================================
-- MIMIC-III Data Extraction and Preprocessing Queries
-- ==============================================================

-- ==============================================================
-- 1. Extract Patient Demographics and Admission Information
-- ==============================================================
-- Purpose:
-- Retrieve distinct patient demographics and admission details for patients diagnosed with sepsis (ICD9 codes '99591' and '99592').
-- Calculates the age of the patient at the time of admission.

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
    a.ETHNICITY,
    DATE_DIFF(DATE(a.ADMITTIME), DATE(p.DOB), YEAR) AS AGE_AT_ADMISSION
FROM
    mimiciii.patients p
JOIN
    mimiciii.diagnoses_icd d
    ON p.SUBJECT_ID = d.SUBJECT_ID
JOIN
    mimiciii.admissions a
    ON d.HADM_ID = a.HADM_ID
WHERE
    d.ICD9_CODE IN ('99591', '99592');

-- ==============================================================
-- 2. Calculate Mean Weight for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average weight (in kg) for each patient diagnosed with sepsis.
-- Utilizes specific ITEMIDs corresponding to weight measurements.

SELECT 
    C.SUBJECT_ID, 
    AVG(C.VALUENUM) AS WEIGHT_MEAN, 
    ITEM.LABEL, 
    ITEM.ITEMID
FROM 
    mimiciii.chartevents AS C
INNER JOIN 
    mimiciii.diagnoses_icd AS D 
    ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN 
    mimiciii.d_items AS ITEM 
    ON ITEM.ITEMID = C.ITEMID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND C.ITEMID IN (226531, 763, 224639, 226512)
GROUP BY 
    C.SUBJECT_ID, ITEM.LABEL, ITEM.ITEMID
ORDER BY 
    C.SUBJECT_ID ASC;

-- ==============================================================
-- 3. Calculate Mean Height for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average height (in cm) for each patient diagnosed with sepsis.
-- Utilizes specific ITEMIDs corresponding to height measurements.

SELECT 
    C.SUBJECT_ID, 
    AVG(C.VALUENUM) AS HEIGHT_MEAN, 
    ITEM.LABEL, 
    ITEM.ITEMID
FROM 
    mimiciii.chartevents AS C
INNER JOIN 
    mimiciii.diagnoses_icd AS D 
    ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN 
    mimiciii.d_items AS ITEM 
    ON ITEM.ITEMID = C.ITEMID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND C.ITEMID IN (226707, 226730, 1394)
GROUP BY 
    C.SUBJECT_ID, ITEM.LABEL, ITEM.ITEMID
ORDER BY 
    C.SUBJECT_ID ASC;

-- ==============================================================
-- 4. Calculate Length of Stay (LOS) in Hospital
-- ==============================================================
-- Purpose:
-- Calculate the length of stay (in days) for each hospital admission.

-- Note:
-- This step is typically performed in a data processing environment (e.g., Python/Pandas)
-- rather than directly in SQL. Below is a placeholder to indicate where this calculation occurs.

-- Python Code Snippet (For Reference Only):
-- 
-- # 1. Ensure 'ADMITTIME' and 'DISCHTIME' are in datetime format
-- patient_df['ADMITTIME'] = pd.to_datetime(patient_df['ADMITTIME'])
-- patient_df['DISCHTIME'] = pd.to_datetime(patient_df['DISCHTIME'])
--
-- # 2. Calculate Length of Stay (LOS) in days
-- patient_df['LOS'] = (patient_df['DISCHTIME'] - patient_df['ADMITTIME']).dt.days

-- ==============================================================
-- 5. Calculate Mean Length of Stay (LOS) in ICU for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average length of stay (in days) in the Intensive Care Unit (ICU) for each patient diagnosed with sepsis.

SELECT 
    ICU.SUBJECT_ID, 
    AVG(ICU.LOS) AS LOS_ICU_MEAN
FROM 
    mimiciii.icustays AS ICU
INNER JOIN 
    mimiciii.diagnoses_icd AS D 
    ON D.SUBJECT_ID = ICU.SUBJECT_ID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND ICU.LOS IS NOT NULL
GROUP BY 
    ICU.SUBJECT_ID
ORDER BY 
    ICU.SUBJECT_ID ASC;

-- ==============================================================
-- 6. Extract Unique Subject IDs with Specific ICD9 Codes
-- ==============================================================
-- Purpose:
-- Retrieve distinct SUBJECT_IDs of patients who have diagnoses matching a wide range of ICD9 codes, specifically those listed, and who also have sepsis diagnoses ('99591', '99592').

SELECT DISTINCT
    subject_id
FROM
    `physionet-data-435019.mimiciii.diagnoses_icd`
WHERE
    (
        icd9_code IN ("2535","3572","5881","7751","24900","24901","24910","24911","24920","24921","24930","24931","24940","24941","24950","24951","24960","24961","24970","24971","24980","24981","24990","24991","25000","25001","25002","25003","25010","25011","25012","25013","25020","25021","25022","25023","25030","25031","25032","25033","25040","25041","25042","25043","25050","25051","25052","25053","25060","25061","25062","25063","25070","25071","25072","25073","25080","25081","25082","25083","25090","25091","25092","25093","64800","64801","64802","64803","64804","V1221","V180","V771")
    )
    AND subject_id IN (
        SELECT subject_id
        FROM `physionet-data-435019.mimiciii.diagnoses_icd`
        WHERE icd9_code IN ('99591', '99592') -- Sepsis codes
    )
ORDER BY SUBJECT_ID ASC;

-- ==============================================================
-- 7. Calculate Minimum Heartrate for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Determine the minimum heartrate (times/min) recorded for each patient diagnosed with sepsis.

SELECT 
    C.SUBJECT_ID,
    ITEM.LABEL,
    MIN(C.VALUENUM) AS MIN_VALUE
FROM 
    `physionet-data.mimiciii_clinical.chartevents` AS C
INNER JOIN 
    `physionet-data.mimiciii_clinical.diagnoses_icd` AS D 
    ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN 
    `physionet-data.mimiciii_clinical.d_items` AS ITEM 
    ON ITEM.ITEMID = C.ITEMID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND C.ITEMID IN (211, 220045)
GROUP BY 
    C.SUBJECT_ID, ITEM.LABEL
ORDER BY 
    C.SUBJECT_ID ASC;

-- ==============================================================
-- 8. Calculate Average Heartrate for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average heartrate (times/min) for each patient diagnosed with sepsis.

SELECT 
    C.SUBJECT_ID,
    ITEM.LABEL,
    AVG(C.VALUENUM) AS AVG_VALUE
FROM 
    `physionet-data.mimiciii_clinical.chartevents` AS C
INNER JOIN 
    `physionet-data.mimiciii_clinical.diagnoses_icd` AS D 
    ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN 
    `physionet-data.mimiciii_clinical.d_items` AS ITEM 
    ON ITEM.ITEMID = C.ITEMID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND C.ITEMID IN (211, 220045)
GROUP BY 
    C.SUBJECT_ID, ITEM.LABEL
ORDER BY 
    C.SUBJECT_ID ASC;

-- ==============================================================
-- 9. Calculate Mean Systolic Blood Pressure (SBP) for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average systolic blood pressure (mmHg) for each patient diagnosed with sepsis.

SELECT 
    C.SUBJECT_ID, 
    AVG(C.VALUENUM) AS SBP_MEAN
FROM 
    mimiciii.chartevents AS C
INNER JOIN 
    mimiciii.diagnoses_icd AS D 
    ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN 
    mimiciii.d_items AS ITEM 
    ON ITEM.ITEMID = C.ITEMID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND C.ITEMID IN (51, 422, 455, 6701, 220050, 220179, 225309)
    AND C.VALUENUM IS NOT NULL
GROUP BY 
    C.SUBJECT_ID
ORDER BY 
    C.SUBJECT_ID ASC;

-- ==============================================================
-- 10. Calculate Mean Diastolic Blood Pressure (DBP) for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average diastolic blood pressure (mmHg) for each patient diagnosed with sepsis.

SELECT 
    C.SUBJECT_ID, 
    AVG(C.VALUENUM) AS DBP_MEAN
FROM 
    mimiciii.chartevents AS C
INNER JOIN 
    mimiciii.diagnoses_icd AS D 
    ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN 
    mimiciii.d_items AS ITEM 
    ON ITEM.ITEMID = C.ITEMID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND C.ITEMID IN (8368, 8441, 8555, 220051, 220180, 225310)
    AND C.VALUENUM IS NOT NULL
GROUP BY 
    C.SUBJECT_ID
ORDER BY 
    C.SUBJECT_ID ASC;

-- ==============================================================
-- 11. Calculate Mean Mean Arterial Pressure (MAP) for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average mean arterial pressure (mmHg) for each patient diagnosed with sepsis.

SELECT 
    C.SUBJECT_ID, 
    AVG(C.VALUENUM) AS MAP_MEAN
FROM 
    mimiciii.chartevents AS C
INNER JOIN 
    mimiciii.diagnoses_icd AS D 
    ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN 
    mimiciii.d_items AS ITEM 
    ON ITEM.ITEMID = C.ITEMID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND C.ITEMID IN (52, 456, 6702, 220052, 220181, 225312)
    AND C.VALUENUM IS NOT NULL
GROUP BY 
    C.SUBJECT_ID
ORDER BY 
    C.SUBJECT_ID ASC;

-- ==============================================================
-- 12. Calculate Mean Respiratory Rate (RR) for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average respiratory rate (times/min) for each patient diagnosed with sepsis.

SELECT 
    C.SUBJECT_ID, 
    AVG(C.VALUENUM) AS RR_MEAN
FROM 
    mimiciii.chartevents AS C
INNER JOIN 
    mimiciii.diagnoses_icd AS D 
    ON D.SUBJECT_ID = C.SUBJECT_ID
JOIN 
    mimiciii.d_items AS ITEM 
    ON ITEM.ITEMID = C.ITEMID
WHERE 
    D.ICD9_CODE IN ('99591', '99592') 
    AND C.ITEMID IN (618, 224422, 224689, 224690, 220210)
    AND C.VALUENUM IS NOT NULL
GROUP BY 
    C.SUBJECT_ID
ORDER BY 
    C.SUBJECT_ID ASC;

-- ==============================================================
-- 13. Calculate Temperature Metrics (Mean, Min, Max) in Celsius for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average, minimum, and maximum body temperature (°C) for each patient diagnosed with sepsis.
-- Converts temperatures from Fahrenheit to Celsius where applicable.

WITH Temperature_Converted AS (
    SELECT
        C.SUBJECT_ID,
        C.CHARTTIME,
        -- Convert Fahrenheit to Celsius; leave Celsius measurements as is
        CASE 
            WHEN C.ITEMID IN (678, 679, 223761) THEN (C.VALUENUM - 32) * 5/9
            ELSE C.VALUENUM
        END AS TEMP_C
    FROM 
        mimiciii.chartevents AS C
    INNER JOIN 
        mimiciii.diagnoses_icd AS D 
        ON D.SUBJECT_ID = C.SUBJECT_ID
    WHERE 
        D.ICD9_CODE IN ('99591', '99592') 
        AND C.ITEMID IN (676, 677, 678, 679, 223762, 223761)
        AND C.VALUENUM IS NOT NULL
),

Temperature_Deduplicated AS (
    SELECT
        SUBJECT_ID,
        TIMESTAMP_TRUNC(CHARTTIME, HOUR) AS CHARTTIME_HOUR,
        AVG(TEMP_C) AS TEMP_C_Avg
    FROM 
        Temperature_Converted
    GROUP BY 
        SUBJECT_ID, CHARTTIME_HOUR
)

SELECT
    SUBJECT_ID,
    AVG(TEMP_C_Avg) AS TEMP_MEAN_C,
    MIN(TEMP_C_Avg) AS TEMP_MIN_C,
    MAX(TEMP_C_Avg) AS TEMP_MAX_C
FROM 
    Temperature_Deduplicated
GROUP BY 
    SUBJECT_ID
ORDER BY 
    SUBJECT_ID ASC;

-- ==============================================================
-- 14. Calculate Oxygen Saturation Metrics (Mean, Min, Max) for Sepsis Patients
-- ==============================================================
-- Purpose:
-- Compute the average, minimum, and maximum oxygen saturation (%) for each patient diagnosed with sepsis.

WITH Oxygen_Saturation_Converted AS (
    SELECT
        C.SUBJECT_ID,
        C.CHARTTIME,
        C.ITEMID,
        C.VALUENUM AS OXYGEN_SAT
    FROM 
        mimiciii.chartevents AS C
    INNER JOIN 
        mimiciii.diagnoses_icd AS D 
        ON D.SUBJECT_ID = C.SUBJECT_ID
    WHERE 
        D.ICD9_CODE IN ('99591', '99592') 
        AND C.ITEMID IN (646, 834, 220227, 220277)
        AND C.VALUENUM IS NOT NULL
),

Oxygen_Saturation_Deduplicated AS (
    SELECT
        SUBJECT_ID,
        TIMESTAMP_TRUNC(CHARTTIME, MINUTE) AS CHARTTIME_MINUTE,
        AVG(OXYGEN_SAT) AS OXYGEN_SAT_Avg
    FROM 
        Oxygen_Saturation_Converted
    GROUP BY 
        SUBJECT_ID, CHARTTIME_MINUTE
)

SELECT
    SUBJECT_ID,
    AVG(OXYGEN_SAT_Avg) AS OXYGEN_SAT_MEAN,
    MIN(OXYGEN_SAT_Avg) AS OXYGEN_SAT_MIN,
    MAX(OXYGEN_SAT_Avg) AS OXYGEN_SAT_MAX
FROM 
    Oxygen_Saturation_Deduplicated
GROUP BY 
    SUBJECT_ID
ORDER BY 
    SUBJECT_ID ASC;
