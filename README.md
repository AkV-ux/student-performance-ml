# Student Performance Prediction using Machine Learning

## Overview
This project predicts whether a student will **pass or fail** based on academic and behavioral factors using a machine learning classification model.

The goal of this project is not just prediction, but to understand:
- how data affects model performance
- how to properly evaluate ML models
- how to interpret model decisions

---

## Dataset
The dataset contains information about students with the following features:

- **study_hours**: Average hours studied per day  
- **attendance**: Attendance percentage  
- **previous_score**: Score in the previous exam  
- **assignments_completed**: Number of assignments completed  
- **pass** (target):  
  - `1` → Pass  
  - `0` → Fail  

Each row represents one student.

---

## Technologies Used
- Python  
- Pandas  
- Scikit-learn  
- Logistic Regression  

---

## Project Phases

### Phase 1: Data Understanding
- Created a structured dataset
- Loaded and inspected data using Pandas
- Verified data types and basic statistics

### Phase 2: Model Training
- Split data into features (X) and target (y)
- Trained a Logistic Regression classifier
- Made predictions on unseen data

### Phase 3: Model Evaluation
- Evaluated model using:
  - Accuracy
  - Confusion Matrix
  - Precision
  - Recall
- Observed evaluation edge cases due to small dataset size
- Learned limitations of random train-test splits on small data

### Phase 4: Feature Importance
- Extracted Logistic Regression coefficients
- Identified which features most influence pass/fail prediction
- Interpreted results for explainability

---

## Key Learnings
- Accuracy alone is not sufficient to evaluate a model
- Small datasets can produce misleading metrics
- Data distribution strongly affects evaluation results
- Model interpretability is as important as performance

---

## Limitations
- Dataset size is small and not suitable for real-world deployment
- Results are sensitive to train-test split
- Model performance would improve with more diverse data

---

## Future Improvements
- Use a larger, real-world dataset
- Apply stratified sampling or cross-validation
- Add visualizations for feature importance
- Experiment with other classification models
