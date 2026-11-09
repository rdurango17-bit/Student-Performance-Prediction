# Student Performance Prediction System

## Problem
The prototype explores whether historical student information can help identify learners who may need additional academic support.

## Dataset
UCI Machine Learning Repository — Student Performance:
https://archive.ics.uci.edu/dataset/320/student+performance

The laboratory uses Mathematics records. The model creates its target from the final Mathematics grade:
- `G3 < 10` = `AT RISK` (class 1)
- `G3 >= 10` = `NOT AT RISK` (class 0)

**G3 is the target only. It is excluded from the model input features to prevent target leakage.**

## Technologies
Python, Pandas, scikit-learn, Random Forest, Streamlit, Joblib, GitHub.

## Installation
Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Train the model
Run this command in the project folder:

```bash
python train_model.py
```

The script retrieves the dataset and creates:
- `student_data.csv`
- `student_model.pkl`

Record the accuracy, precision, recall, and F1-score printed in the terminal.

## Run the Streamlit app

```bash
streamlit run app.py
```

Open the local address shown in the terminal (usually http://localhost:8501). Download the sample input CSV from the app, upload it, review the predictions, and download the results.

## SAD mini-analysis
- **Problem:** Academic risk may be noticed only after poor results. The prototype explores earlier identification using available student indicators.
- **Users:** Faculty member or academic analyst.
- **Inputs:** Student attributes and academic indicators such as study time, absences, past failures, G1, and G2.
- **Process:** Validate data → preprocess → apply trained model → classify risk.
- **Outputs:** AT RISK / NOT AT RISK label and model evaluation summary.
- **Limitation:** A model trained on a Portuguese secondary-school dataset may not generalize to another institution or student population.

## Mini use case
The user uploads a CSV file → the system checks required columns → the model predicts risk → the system displays and exports the results.

## Testing checklist
- [ ] `train_model.py` creates the CSV and model files.
- [ ] Streamlit dashboard opens.
- [ ] Uploading a valid sample CSV displays predictions.
- [ ] Uploading a CSV with a missing required column displays an error.
- [ ] Prediction results can be downloaded.
- [ ] GitHub repository contains the source files and README.

## Responsible use
This is a classroom demonstration using public, historical data. Predictions are not diagnoses and must not be used as the sole basis for real student decisions. A faculty member should verify the student's actual records and discuss appropriate support with the student.

## GitHub commands
Create a public repository named `student-performance-prediction`, then run:

```bash
git init
git add train_model.py app.py requirements.txt README.md
git commit -m "Build student performance prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Never place passwords or access tokens in source code.
