import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL.GimpGradientFile import linear
from sklearn.metrics.pairwise import rbf_kernel

df = pd.read_csv("diabetes.csv")
print(df.head(5))

print(df.columns)

print(df.isnull().sum())
print(df.duplicated().sum())

# ML MODEL TRAINING
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


print(df.corr())
#sns.heatmap(df.corr())
#plt.show()

X = df.drop("Outcome",axis=1)
y = df["Outcome"]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
# Using Logistic Regression Model
model = LogisticRegression(max_iter=10000)
print(model.fit(X_train,y_train))
y_pred = model.predict(X_test)
print(y_pred)

# Accuracy
acc = accuracy_score(y_test,y_pred)
print(acc)

# Using Random Forest Model
rf_model = RandomForestClassifier(n_estimators=10000)
print(rf_model.fit(X_train,y_train))
y_pred_rf = rf_model.predict(X_test)
print(y_pred_rf)

rf_acc = accuracy_score(y_test,y_pred_rf)
print(rf_acc)


# SVM Support Vector Machine
pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear')
)

param_grid = {
    'svc__C':[0.1,1,10],
    'svc__kernel':['linear','poly'],
    'svc__gamma':['scale',0.01,0.001]
}

grid = GridSearchCV(pipeline,param_grid,cv=5,scoring='accuracy',verbose=0)
print(grid.fit(X_train,y_train))

svc_y_pred = grid.predict(X_test)
print(svc_y_pred)

print(grid.best_params_)
print(accuracy_score(y_test,svc_y_pred))
print(classification_report(y_test,svc_y_pred))


cm = confusion_matrix(y_test,svc_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=grid.classes_)
disp.plot(cmap=plt.cm.Blues)
print(plt.show())



# plot the classes of target data
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the class distribution of target data (y_train or y_test)
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train)
plt.title("Class Distribution in Target Variable (y_train)")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svm = SVC(kernel='linear')
svm.fit(X_train_pca,y_train)


import numpy as np
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Paired)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='Set1', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Support Vector Machine (SVM) Decision Boundary')
plt.show()

import numpy as np
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Paired)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='Set1', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Support Vector Machine (SVM) Decision Boundary')
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Adaboost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
ada = AdaBoostClassifier(n_estimators=100,random_state=42)
ada.fit(X_train,y_train)
y_pred = ada.predict(X_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))

# Define models
from sklearn.ensemble import HistGradientBoostingClassifier

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
    "Stacking": StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svc', SVC(probability=True, random_state=42)),
            ('hgb', HistGradientBoostingClassifier(max_iter=100, random_state=42))
        ],
        final_estimator=LogisticRegression()
    )
}

# Evaluate all models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"{name} Accuracy: {acc:.4f}")

# Display as DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
print("\nModel Comparison:")
print(results_df)


# Plotting accuracy comparison from results_df
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.ylim(0, 1)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# dump the ML models using 'pickle' and 'joblib'

import pickle
# save model
with open('xgb_model.pkl','wb') as f:
  pickle.dump(model,f)

with open('xgb_model.pkl','rb') as f:
  model = pickle.load(f)

import joblib

joblib.dump(model, 'xgb_model.joblib')
model = joblib.load('xgb_model.joblib')








