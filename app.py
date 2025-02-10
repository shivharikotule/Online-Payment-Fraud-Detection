import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score,f1_score, roc_auc_score


df=pd.read_csv("Fraud_data.csv")

df.head()

df.tail()

df.info()

df.describe()

df.isnull().sum()

df.shape

fraud_df = df[df['isFraud'] == 1]  # fraud cases
non_fraud_df = df[df['isFraud'] == 0]  # non-fraud cases

# Randomly sample % of non-fraud cases
non_fraud_sample = non_fraud_df.sample(frac=0.028, random_state=42)

# Combine fraud and sampled non-fraud data
balanced_df = pd.concat([fraud_df, non_fraud_sample])

balanced_df.shape

balanced_df["isFraud"].value_counts()

balanced_df.columns

balanced_df.drop(['nameOrig', 'nameDest','isFlaggedFraud'], axis=1, inplace=True)

balanced_df.columns

balanced_df['type'].head(10)

balanced_df['type'].replace({'CASH_OUT':0, 'PAYMENT':1, 'CASH_IN':2, 'TRANSFER':3, 'DEBIT':4}, inplace=True)

balanced_df['type'].value_counts()

balanced_df.head()

# X Data
x = balanced_df.drop(['isFraud'], axis=1)
print('X shape is : ' , x.shape)
print()

# y Data
y = balanced_df['isFraud']
print('Y shape is : ' , y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

Model_LR = LogisticRegression(class_weight='balanced')
Model_LR.fit(X_train, y_train)


y_pred_LR = Model_LR.predict(X_test)


Train_Accuracy = Model_LR.score(X_train, y_train)
Test_Accuracy = Model_LR.score(X_test, y_test)
print(f'Training accuracy: {Train_Accuracy*100:.2f} %')
print(f'Testing accuracy: {Test_Accuracy*100:.2f} %')

print(classification_report(y_test,y_pred_LR))

print(confusion_matrix(y_test,y_pred_LR))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_samples = 0.8, oob_score = True,
                       random_state=7, class_weight='balanced')

rf.fit(X_train, y_train)



y_pred_rf = rf.predict(X_test)


Train_Accuracy = rf.score(X_train, y_train)
Test_Accuracy = rf.score(X_test, y_test)
print(f'Training accuracy: {Train_Accuracy*100:.2f} %')
print(f'Testing accuracy: {Test_Accuracy*100:.2f} %')

print(classification_report(y_test,y_pred_rf))

print(confusion_matrix(y_test,y_pred_rf))

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(X_train, y_train)


y_pred_dt = dt.predict(X_test)


Train_Accuracy = dt.score(X_train, y_train)
Test_Accuracy = dt.score(X_test, y_test)
print(f'Training accuracy: {Train_Accuracy*100:.2f} %')
print(f'Testing accuracy: {Test_Accuracy*100:.2f} %')

print(classification_report(y_test,y_pred_dt))

print(confusion_matrix(y_test,y_pred_dt))

Recall_LR = recall_score(y_test, y_pred_LR)

Recall_rf = recall_score(y_test, y_pred_rf)

Recall_dt = recall_score(y_test, y_pred_dt)

evaluation = pd.DataFrame({'Classification Model': ['Logistic Regression','Decision Tree', 'Random Forest'],
                           'Accuracy Rate': [(Recall_LR*100).round(2), (Recall_dt*100).round(2), (Recall_rf*100).round(2)]})

evaluation

