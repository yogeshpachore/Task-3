# Predicting University Admissions

## Task 3: Advanced Outlier Detection and Treatment

### Objective
Develop and apply an outlier detection and visualization pipeline across the university admission applicant dataset to systematically identify, compare, visualize, and treat anomalous observations using boxplots, scatterplots, and statistical methods before machine learning model development.

### Dataset
The analysis uses `Admission_Predict.csv`.

The dataset contains 400 applicant records and 9 columns:
- `Serial No.` — Applicant identifier
- `GRE Score`
- `TOEFL Score`
- `University Rating`
- `SOP`
- `LOR`
- `CGPA`
- `Research`
- `Chance of Admit`

`Serial No.` is an identifier and is removed from the analysis because it is not an applicant feature.

### Analysis Performed
1. **Dataset Preparation** — loaded the admission dataset and created an analysis dataframe after removing the applicant identifier.
2. **Numerical Feature Identification** — identified the numerical variables used for outlier visualization and statistical analysis.
3. **Boxplot Analysis** — generated boxplots to visually inspect distributions and identify potential unusual observations.
4. **Scatterplot Analysis** — examined feature relationships using scatterplots, including GRE Score, TOEFL Score, CGPA, and Chance of Admit.
5. **IQR Outlier Detection** — calculated Q1, Q3, IQR, lower bounds, upper bounds, and outlier counts using the 1.5 × IQR rule.
6. **Z-Score Outlier Detection** — identified potential outliers using the criterion `|Z| > 3`.
7. **Method Comparison** — compared the number of observations detected by the IQR and Z-score methods.
8. **Outlier Record Inspection** — examined detected outlier records, including the identified CGPA outlier.
9. **Outlier Treatment** — applied IQR-based capping to selected continuous applicant features: GRE Score, TOEFL Score, University Rating, SOP, LOR, and CGPA.
10. **Treatment Verification** — compared the original and treated data and verified the number of values affected by capping.

### Key Findings
The IQR analysis identified a small number of potential outliers. LOR and CGPA each contained one IQR-based outlier, while Chance of Admit contained two potential IQR outliers.

The Z-score analysis identified one outlier in CGPA and did not identify outliers in the other analyzed variables using the `|Z| > 3` threshold.

The comparison of IQR and Z-score results demonstrates that different statistical methods can identify different observations as potential outliers.

During treatment, one LOR value and one CGPA value were capped using the IQR boundaries. The treatment preserved the dataset size because values were capped instead of deleting complete records.

`Research` is a binary variable and `Chance of Admit` is the target variable, so they were not included in the selected continuous-feature treatment process.

### Conclusion
Task 3 provides a focused outlier analysis and treatment pipeline for the university admission dataset. Boxplots and scatterplots are used for visual investigation, while IQR and Z-score methods provide statistical outlier detection.

The detected anomalies are investigated rather than automatically removed. IQR capping is then applied to selected continuous applicant features to reduce the influence of extreme observations while preserving all 400 records.

This preprocessing step provides a cleaner dataset for subsequent machine learning model development.

### Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook

### File
`Task-3(1).ipynb`
