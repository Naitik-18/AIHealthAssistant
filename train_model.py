import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# 1. Load dataset
df = pd.read_csv('data/disease_symptoms.csv')
print("Raw dataset sample:")
print(df.head())

# 2. Get all unique symptom names (drop NaN)
symptom_cols = [col for col in df.columns if col.startswith('Symptom')]
all_symptoms = sorted(
    set(
        s.strip()
        for col in symptom_cols
        for s in df[col].dropna().unique()
    )
)

print(f"✅ Found {len(all_symptoms)} unique symptoms")

# 3. One-hot encode symptoms (0/1)
one_hot_data = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for idx, row in df.iterrows():
    for col in symptom_cols:
        sym = row[col]
        if pd.notna(sym):
            sym_clean = sym.strip()
            one_hot_data.loc[idx, sym_clean] = 1

# 4. X = symptom binary matrix, y = disease labels
X = one_hot_data
y = df['Disease']

# 5. Encode disease labels into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("✅ Diseases:", list(le.classes_))

# 6. Train/test split (stratify keeps same class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 7. Train a Random Forest (handles imbalance better)
clf = RandomForestClassifier(
    n_estimators=200,         # more trees for stable predictions
    max_depth=15,             # slightly deeper trees
    class_weight="balanced_subsample",  # better balance for rare classes
    random_state=42,
    n_jobs=-1                 # use all CPU cores
)
clf.fit(X_train, y_train)

# 8. Evaluate
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"\n✅ Accuracy: {accuracy:.2%}")

# Avoid division-by-zero warnings
print("\n📊 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    zero_division=0
))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save trained model + label encoder + symptom list
with open('model/dt_model.pkl', 'wb') as f:
    pickle.dump({
        'model': clf,
        'label_encoder': le,
        'symptoms': all_symptoms
    }, f)

print("\n✅ Model trained & saved successfully as model/dt_model.pkl")
