# Student Performance/Support Prediction Model

A machine learning project that predicts whether students need academic support based on their test scores and demographic information using logistic regression.

## Overview

This project analyzes student performance data to identify students who may need additional academic support. The model uses various features including test scores, demographics, and educational background to make predictions.

## Features

- **Data Processing**: Calculates average scores from math, reading, and writing tests
- **Support Classification**: Automatically identifies students needing support (average score < 60)
- **Categorical Encoding**: Handles non-numeric data for machine learning compatibility
- **Model Training**: Uses logistic regression for binary classification
- **Performance Evaluation**: Provides accuracy metrics and confusion matrix visualization

## Dataset Requirements

The model expects a CSV file named `test_data.csv` with the following columns:

- `math score` - Student's math test score
- `reading score` - Student's reading test score  
- `writing score` - Student's writing test score
- `gender` - Student's gender
- `race/ethnicity` - Student's racial/ethnic background
- `parental level of education` - Parent's education level
- `lunch` - Lunch program participation (standard/free or reduced)
- `test preparation course` - Whether student completed test prep course

## Dependencies

```python
pandas
scikit-learn
matplotlib
```

Install dependencies:
```bash
pip install pandas scikit-learn matplotlib
```

## How It Works

1. **Data Loading**: Reads student data from CSV file
2. **Feature Engineering**: Calculates average test scores
3. **Target Creation**: Labels students as needing support if average < 60
4. **Data Preprocessing**: Encodes categorical variables using LabelEncoder
5. **Model Training**: Splits data (50/50) and trains logistic regression model
6. **Evaluation**: Generates accuracy score and confusion matrix
7. **Visualization**: Displays confusion matrix heatmap

## Usage

1. Ensure your dataset is named `test_data.csv` and placed in the same directory
2. Run the script:
   ```bash
   python student_support_model.py
   ```

## Output

The model provides:
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: Detailed breakdown of correct/incorrect predictions
- **Total Predictions**: Number of students evaluated
- **Visualization**: Confusion matrix heatmap

## Model Performance

The logistic regression model uses a 50/50 train-test split with:
- Maximum iterations: 1000
- Random state: 42 (for reproducible results)

## Interpretation

- **True Positives**: Students correctly identified as needing support
- **True Negatives**: Students correctly identified as not needing support  
- **False Positives**: Students incorrectly flagged as needing support
- **False Negatives**: Students who need support but weren't identified

## Future Improvements

- Feature scaling for better model performance
- Cross-validation for more robust evaluation
- Additional algorithms (Random Forest, SVM) for comparison
- Feature importance analysis
- Handling of missing data
