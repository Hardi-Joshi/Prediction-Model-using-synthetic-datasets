# Prediction-Model-using-synthetic-datasets
Implementation of a disease prediction model using a synthetic dataset for rare disease. To solve data scarcity problem.
```python
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# 1. Load data
df = pd.read_excel("synthetic_checked_fabry_like.xlsx")

# 2. Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Convert sex
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})

# 4. Combine signs + symptoms
sign_cols = [col for col in df.columns if 'sign' in col]
symptom_cols = [col for col in df.columns if 'symptom' in col]

df['all_features'] = df[sign_cols + symptom_cols].astype(str).agg(' '.join, axis=1)

# 5. TF-IDF
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(df['all_features'])

# 6. Combine features
numerical_features = df[['age', 'sex']].values
X = np.hstack((numerical_features, text_features.toarray()))
y = df['fabry_like']

# 7. Split (inside code only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

# 9. Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")
```
