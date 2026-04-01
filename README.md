# Diabetes-Risk-Evaluation-ML-Project
## Description
A Machine Learning Project that encompasses an end-to-end pipeline to create high-performing models, from an  Imbalanced Dataset of large scale survey of health data, to predict the risk of diabetes.

## Technical Details
- **Dataset**: The dataset used consists of 768 patients diagnosed with diabetes (603 with diabetes, 165 without, and 25 undiagnosed) with 8 features: Pregnancies, Glucose, Blood Pressure, SkinThickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
- **Implemented:** Data analysis, preprocessing, feature engineering, and data sampling strategies.
- **Classification Models:** Random Forest, SVM, and MLP with optimized hyperparameters.
- **Addressed class imbalance problem** through Hybrid Sampling techniques, with improved Recall.
- **Metrics**: A focus on optimizing Recall while minimizing F1 score trade-offs.

## 2. Preprocessing Techniques
Preprocessing techniques applied to the dataset include:
a. Standard Scaling (SS)
b. Min-Max Scaling (MMS)
c. Logarithmic Transformation (LT)
d. Square Root Transformation (SRT)

Impact on Recall for each model in both binary and ternary scenarios:
Binary Scenario (Diabetic vs Non-diabetic):
* SS showed the highest Recall across all models, with an average increase of 6.7%.
* MMS led to a slight improvement in Recall for SVM but had no significant impact on RF or MLP.
* LT and SRT improved Recall for RF and MLP but resulted in lower F1 scores compared to SS.

Ternary Scenario (Diabetic, Non-diabetic, Undiagnosed):
* SS and MMS provided a noticeable boost in recall across all models, with an average increase of 7.8% for SS and 6.1% for MMS.
* LT and SRT resulted in mixed performance, with LT improving Recall for RF but worsening it for SVM and MLP. SRT had little to no impact on any model's Recall.


## 3. Data Sampling Techniques
Data sampling strategies utilized include:
a. Undersampling (US) methods, such as Random Undersampling (RUS), NearMiss undersampling (NMS), and Tomek Links undersampling (TLUS).
b. Hybrid Sampling (HS) approach that combines oversampling with undersampling techniques.

Analysis of the impact on Recall performance across models and scenarios:
Binary Scenario:
* Undersampling methods generally improved Recall for all models, with RUS and NMS achieving an average increase of 9.5%. 
* HS demonstrated the highest Recall improvement (23.6%) compared to the original dataset.

Ternary Scenario:
* US techniques resulted in minimal improvements in Recall across all models, with an average boost of 1.4% for RUS and NMS.
* HS significantly increased Recall, achieving an average improvement of 19.3%.

## 4. Model Performance Comparison
Performance comparison of each model in terms of Recall and F1 scores under various conditions (preprocessing, sampling) for both binary and ternary scenarios:

Binary Scenario:
* SVM showed the highest F1 score on average with various preprocessing techniques, but RF achieved the best balance between Recall and F1 score after applying HS.
* MLP demonstrated the poorest performance among the three models in terms of Recall and F1 scores, regardless of preprocessing or sampling methods used.

Ternary Scenario:
* SVM achieved the highest F1 score across all models, with SS as the best preprocessing method for SVM and HS as the optimal sampling strategy.
* RF showed a better balance between Recall and F1 score compared to MLP after applying HS and SS.

## 5. Trade-offs Between Metrics
Discussion on the performance trade-offs between Recall and F1 scores observed across models and scenarios:

Binary Scenario:
* SVM generally achieved higher F1 scores but lower Recall compared to RF and MLP, making it less suitable for imbalanced datasets with a focus on Recall optimization.
* RF demonstrated better balance between Recall and F1 score in most cases, with the exception of preprocessing using LT and SRT.

Ternary Scenario:
* SVM performed well overall, maintaining high F1 scores while improving Recall after applying HS and SS. However, the trade-off between Recall and F1 score was more significant than in the binary scenario.
* RF showed a better balance between Recall and F1 score compared to MLP, particularly when using HS for sampling and SS for preprocessing.

## 6. Cross-Scenario Pattern Analysis
Identification of consistent patterns and key divergences in model performance across binary and ternary scenarios:

Consistent patterns:
* SVM generally performed well in terms of F1 score but underperformed in Recall, particularly in the binary scenario when using LT and SRT for preprocessing.
* RF demonstrated a better balance between Recall and F1 score compared to MLP in both scenarios.

Key divergences:
* Ternary Scenario showed improved performance across all models after applying HS, particularly in terms of Recall. This is attributed to the increase in the number of cases for the undiagnosed class.
* Binary Scenario demonstrated more significant improvements with various preprocessing techniques and US methods compared to the ternary scenario. This can be explained by the lower complexity of the binary classification problem.

## 7. Optimization and Future Work
Discussion on the optimization of grid search parameters for each model, suggesting further exploration of advanced sampling methods such as SMOTE, ADASYN, or other ensemble techniques to improve Recall performance.

Recommendations for future work include:
- a. Evaluating models using other imbalanced metrics like Precision, Accuracy, and Specificity.
- b. Comparing the current study's findings with other datasets or using additional preprocessing techniques such as SMOTE, ADASYN, or other hybrid methods for data sampling.
- c. Investigating the impact of different hyperparameter configurations on model performance in terms of Recall and F1 scores across various scenarios.

