To view the report and documentation, please use the [repository wiki](https://github.com/Musiik-fn/650-DRAGON-SLAYERS/wiki)

# BANA 650 PROJECT

**Group Members:** Joshua Cabal, Ida Karimi, Viridiana Radillo

## Overview
 The objective is to reproduce experiments in a published paper. The successful resulting project can potentially lead to research publication at health informatics venues such as AMIA, JAMIA, and JBI.

## Chosen Research Paper
**Research Paper Name**: Predicting 30‑days mortality for MIMIC‑III patients with sepsis‑3: a machine learning
approach using XGboost

**Citation**: Hou, N., Li, M., He, L. et al. Predicting 30-days mortality for MIMIC-III patients with sepsis-3: a machine learning approach using XGboost. J Transl Med 18, 462 (2020). https://doi.org/10.1186/s12967-020-02620-5

## Abstract

**Background**: Sepsis is a significant cause of in-hospital mortality, particularly among ICU patients. Early prediction of sepsis is essential, as prompt and appropriate treatment can improve survival outcomes. Machine learning methods are flexible prediction algorithms with potential advantages over conventional regression and scoring systems. The aim of this study were to replicate and validate the findings performed by [Hou et al. (2020)](https://doi.org/10.1186/s12967-020-02620-5), which found that XGBoost performed better than traditional predictive models.

**Methods**: Using the MIMIC-III v1.4 database, we identified patients with sepsis-3. The data were split into two groups based on death or survival within 30 days. The variables used were from the source paper, which selected based on clinical significance and availability through stepwise analysis between groups. Three predictive models were constructed using R software: a conventional logistic regression model, the random forest prediction model, and an XGBoost algorithm model. The performances of the three models were tested and compared using the area under the receiver operating characteristic curve (AUC), average precision through precision-recall curves, and decision curve analysis through net benefit curves. Finally, a clinical impact curve was used to validate the clinical utility model.

**Results**: A total of 4,555 sepsis-3 patients were included in the study, among whom 1,274 patients died and 3,281 survived within 30 days. According to the results of the AUCs—0.75 (95 % CI: 0.72-0.78) for the logistic regression model, 0.77 (95 % CI: 0.73-0.80) for the random forest model, and 0.79 (95 % CI: 0.75-0.82) for the XGBoost model—and average precision, the XGBoost model performed best. The clinical impact curve and the net benefit curves verified that the XGBoost model possesses clinical utility.

**Conclusions**: A more significant predictive model can be built using the machine learning technique XGBoost. This XGBoost model may prove clinically useful and assist clinicians in tailoring management and triage for patients with sepsis-3.
