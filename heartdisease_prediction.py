# Example pipeline (sketch) — run in Colab / local
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, brier_score_loss
import shap

# Load data
df = pd.read_csv('heart.csv')
X = df.drop(columns=['target'])
y = df['target']  # 0/1

# Split (stratified)
X_train, X_hold, y_train, y_hold = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Feature lists
num_feats = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_feats = X_train.select_dtypes(include=['object','category']).columns.tolist()

# Preprocessor
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                     ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preproc = ColumnTransformer([('num', num_pipe, num_feats),
                             ('cat', cat_pipe, cat_feats)])

# Full pipeline
model = Pipeline([
    ('pre', preproc),
    ('clf', XGBClassifier(n_estimators=200, scale_pos_weight= (len(y_train)-y_train.sum())/y_train.sum(),
                          use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Train
model.fit(X_train, y_train)

# Predict & evaluate
y_pred_proba = model.predict_proba(X_hold)[:,1]
auc = roc_auc_score(y_hold, y_pred_proba)
print('AUC:', auc)

# SHAP explainability
explainer = shap.Explainer(model.named_steps['clf'])
X_hold_trans = model.named_steps['pre'].transform(X_hold)
shap_values = explainer(X_hold_trans)
feature_names = model.named_steps['pre'].get_feature_names_out()
shap_values_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_values_df.head()
clean_names = [name.split('__')[-1] for name in feature_names]
shap_values_df.columns = clean_names
shap_values_df.head()

# SHAP summary plots
shap.summary_plot(shap_values, features= X_hold_trans, feature_names=feature_names, plot_type='bar')
shap.summary_plot(shap_values, features= X_hold_trans, feature_names=feature_names)
