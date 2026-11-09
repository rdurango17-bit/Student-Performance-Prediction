import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Retrieve the UCI Student Performance dataset
student = fetch_ucirepo(id=320)
features = student.data.features.copy()
targets = student.data.targets.copy()
df = pd.concat([features, targets], axis=1)

# If the repository response contains a course indicator, retain Mathematics only.
# Otherwise, the returned records are used as provided by the repository interface.
course_columns = [c for c in df.columns if c.lower() in ("course", "subject")]
if course_columns:
    col = course_columns[0]
    df = df[df[col].astype(str).str.lower().isin(["math", "mathematics", "mat"])].copy()

if "G3" not in df.columns:
    raise ValueError("The downloaded dataset does not contain the required G3 target column.")

df = df.drop_duplicates().reset_index(drop=True)
df.to_csv("student_data.csv", index=False)

print("Dataset shape:", df.shape)
print("\\nFirst five rows:")
print(df.head())
print("\\nData types:")
print(df.dtypes)
print("\\nMissing values:")
print(df.isnull().sum())

# 2. Define the binary target. G3 is NEVER used as an input feature.
y = (df["G3"] < 10).astype(int)
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\\nMODEL EVALUATION")
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Not at risk", "At risk"],
    zero_division=0
))

joblib.dump(model, "student_model.pkl")
print("\\nSaved student_data.csv and student_model.pkl")
