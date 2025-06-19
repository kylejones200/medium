import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('public_health_data.csv')
X = data.drop(columns=['Disease_Risk'])
y = data['Disease_Risk']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#SHAP Analysis
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, feature_names=X.columns)

shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[0], feature_names=X.columns)

shap.summary_plot(shap_values[1], X_test, feature_names=X.columns, plot_type='bar')
