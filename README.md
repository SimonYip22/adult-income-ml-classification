# Adult Census Mixed-Data Classification

Machine learning notebook using sklearn and classical machine learning methods

Predict whether an individual's recorded income falls above or below the dataset threshold using demographic and employment-related features.

### Planned notebook sections

1. **Problem definition**
   - prediction target;
   - task type;
   - intended learning purpose;
   - ethical sensitivity of the features and target.

2. **Dataset loading and inspection**
   - shape;
   - feature types;
   - target distribution;
   - missing or unusual values;
   - duplicated records where relevant.

3. **Feature and target definition**
   - numerical columns;
   - categorical columns;
   - target encoding.

4. **Data splitting**
   - stratified train/test split;
   - explanation of why the test set remains untouched during development.

5. **Baseline**
   - `DummyClassifier`;
   - baseline metric interpretation.

6. **Preprocessing**
   - numerical imputation where required;
   - numerical scaling for linear models;
   - categorical imputation;
   - one-hot encoding;
   - `ColumnTransformer`;
   - `Pipeline`.

7. **Models**
   - logistic regression;
   - decision tree;
   - one ensemble model: random forest or histogram gradient boosting.

8. **Validation**
   - cross-validation;
   - consistent metrics across models;
   - comparison of mean and variation across folds.

9. **Limited tuning**
   - one small justified search for the strongest model;
   - no exhaustive leaderboard optimisation.

10. **Final test evaluation**
    - accuracy;
    - precision;
    - recall;
    - F1;
    - confusion matrix;
    - optionally ROC-AUC or PR-AUC if properly explained.

11. **Error analysis**
    - inspect representative incorrect predictions;
    - discuss whether errors cluster around particular subgroups or feature patterns;
    - avoid causal claims.

12. **Fairness and limitations**
    - historical and social bias;
    - sensitive and proxy attributes;
    - limitations of the target;
    - dataset shift;
    - why high predictive accuracy does not justify high-stakes deployment.
