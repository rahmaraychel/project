import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


# =====================
# 1. Extract Data
# =====================
zip_path = r"C:\Users\user\Desktop\project\autism.csv.zip"
extract_dir = r"C:\Users\user\Desktop\project\autism_data"

# Extract the ZIP only once
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Load train/test data
train = pd.read_csv(os.path.join(extract_dir, "train.csv"))
test = pd.read_csv(os.path.join(extract_dir, "test.csv"))

print("Data loaded successfully!")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
# 2. Preprocessing
test_ids = test["ID"]
train = train.drop(columns=["ID"])
test = test.drop(columns=["ID"])

cat_cols = train.select_dtypes(include=["object"]).columns
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

train[cat_cols] = enc.fit_transform(train[cat_cols])
test[cat_cols] = enc.transform(test[cat_cols])

X = train.drop(columns=["Class/ASD"])
y = train["Class/ASD"]

# 3. Train/Validation Split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Model Training

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)


# 5. Evaluation

y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("Cross-Validation Accuracy:", cv_scores.mean(), "±", cv_scores.std())


# 6. Predictions on Test Set

test_preds = model.predict(test)

output = pd.DataFrame({"ID": test_ids, "Class/ASD": test_preds})
output.to_csv(r"C:\Users\user\Desktop\project\autism_predictions.csv", index=False)
print("Predictions saved to C:\\Users\\user\\Desktop\\project\\autism_predictions.csv")


# 7. Save Model & Encoder

joblib.dump(model, r"C:\Users\user\Desktop\project\random_forest_model.pkl")
joblib.dump(enc, r"C:\Users\user\Desktop\project\ordinal_encoder.pkl")
print(" Model and encoder saved!")

wcss = []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    #elbow geaph
    sns.set()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
