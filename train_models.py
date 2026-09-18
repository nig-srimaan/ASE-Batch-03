import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

print("Generating synthetic network traffic dataset...")
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)

print("Training Logistic Regression (Fastest, Lowest CPU, Lowest Accuracy)...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)

print("Training Random Forest (Balanced)...")
rf = RandomForestClassifier(n_estimators=50, max_depth=5)
rf.fit(X, y)

print("Training XGBoost (Slowest, Highest CPU, Highest Accuracy)...")
xgb = XGBClassifier(n_estimators=100, max_depth=10, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X, y)

# Save models to disk
joblib.dump(lr, 'model_logistic.pkl')
joblib.dump(rf, 'model_rf.pkl')
joblib.dump(xgb, 'model_xgboost.pkl')
print("All models trained and saved successfully!")
