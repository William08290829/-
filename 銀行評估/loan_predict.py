import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core import indexing

df = pd.read_csv("loan_prediction_training_data.csv")

df.head()
df.describe()
df['Credit_History'].value_counts()     #1.0->475, 0.0->89
df['Self_Employed'].value_counts()
df['Education'].value_counts()
df.info()

df['ApplicantIncome'].hist(bins=100)
df['ApplicantIncome'].value_counts()

df.boxplot(column='ApplicantIncome', by='Education')

df.boxplot(column='LoanAmount', by='Education')

df.boxplot(column='ApplicantIncome', by='Self_Employed')


temp2 = df.pivot_table(values='Loan_Status', index=['Credit_History'], aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

temp5 = pd.crosstab(df['Credit_History'],df['Loan_Status'])
temp5.plot(kind='bar', stacked=True, color=['red','blue'],grid=False)

temp6 = pd.crosstab([df['Credit_History'],df['Gender']],df['Loan_Status'])
temp6.plot(kind='bar',stacked=True, color=['red','blue'])


df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
df['TotalIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df.apply(lambda x: sum(x.isnull()),axis=0)

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education',
'Self_Employed','Property_Area','Loan_Status']

df['Self_Employed'].value_counts()
print(500/(500+82))
df['Self_Employed'].fillna("No", inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
df.apply(lambda x: sum(x.isnull()),axis=0)
df['LoanAmount'].value_counts()

table = df.pivot_table(values='LoanAmount',index='Self_Employed',columns='Education',aggfunc=np.median)
def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]
df['LoanAmount'].fillna(df.apply(fage, axis=1),inplace=True)

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df['TotalIncome'] = df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
print(df['TotalIncome_log'].hist(bins=20))

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df.apply(lambda x: sum(x.isnull()),axis=0)

df.dtypes
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education',
'Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
def loan_model(model, data, predictors, outcome, t_size, rs_number):
    X = data[predictors]
    y = data[outcome]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=rs_number)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    print(f"Accuracy:{accuracy}")
    print(f"Recall:{recall}")
    print(f"Precision:{precision}")

#1
outcome_var ='Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
loan_model(model, df, predictor_var, outcome_var, 0.3, 6)