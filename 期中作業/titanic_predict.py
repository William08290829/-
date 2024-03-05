from numpy.core.numeric import full
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df_org = pd.read_csv("train_data_titanic.csv")
# df.head()
# df.info()
# Age Cabin Embarked 資料有缺失
# drop 更多值
df = df_org.drop(['Name', 'Ticket', 'Parch', 'SibSp'], axis=1) #axis 是指前面的資料是直的還是橫的

# sns.pairplot(df[['Survived', 'Fare']], dropna=True)

#用Survived的角度看 (平均)
df.groupby('Survived').mean()
df['Survived'].value_counts()
df_org['SibSp'].value_counts()
df_org['Parch'].value_counts()
# df['Sex'].value_counts() #male 577  female 314

#Handle missing values
df.isnull().sum() #跟總數差多少
len(df)
len(df)/2
df.isnull().sum()>(len(df)/2) #哪些少於資料的總數的一半

#處理Cabin
df.drop('Cabin',axis=1,inplace=True)

#處理Age
df['Age'].isnull().value_counts()
# df.groupby('Sex')['Age'].median().plot(kind='bar')
#補掉缺失的值
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))
df['Age'].isnull().value_counts()

#補Embarked
df['Embarked'].value_counts() #S最多
df['Embarked'].value_counts().idxmax() #S
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)
df['Embarked'].value_counts() #缺的值補上S 644+2= 646

df.isnull().sum()   #所有缺失值補回

df = pd.get_dummies(data=df, columns=['Sex','Embarked'])    #Sex轉換成是否爲男生、是否爲女生，Embarked轉換爲是否爲S、是否爲C、是否爲Q
# df.head()
df.drop(['Sex_female'], axis=1, inplace=True)
# df.head()

sibDf = pd.DataFrame()
sibDf['SiblingNumber'] = df_org['SibSp'] + 1
sibDf['Sib_few'] = sibDf['SiblingNumber'].map(lambda s:1 if s <= 6 else 0)
sibDf['Sib_many'] = sibDf['SiblingNumber'].map(lambda s:1 if 6 < s else 0)
sibDf['Survived'] = df_org['Survived']
sibDf.head()
sibDf.drop(['SiblingNumber','Sib_few'],axis=1,inplace=True)
sibDf.head()
# print(sibDf['Sib_many'].value_counts())
# sns.pairplot(sibDf[['Sib_many','Survived']], dropna=True)


# model1_train = sibDf.drop(['Sib_many','SiblingNumber'],axis=1)
# model2_train = df['Survived']
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# lr.fit(model1_train, model2_train)
# predictions = lr.predict(model1_train)
# #Evaluate
# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
# accuracy_score(model2_train, predictions)
# recall_score(model2_train, predictions)
# precision_score(model2_train, predictions)
# pd.DataFrame(confusion_matrix(model2_train, predictions), columns=['Predict not Survived', 'Predict Survived'], index=['True not Survived', 'True Survived'])

# df['Name'].head()
# def getName(name):
#     str1 = name.split(",")[0]
#     str2 = str1.strip()
#     return str2
# def findT(name):
#     return name.find('t' or 'T')
# nameDf = pd.DataFrame()
# nameDf['Name'] = df['Name'].map(getName)
# nameDf.head()


df.corr()
#Prepare training data
#把Survived, Pclass丟掉
X = df.drop(['Survived','Pclass'],axis=1)
y = df['Survived']
# df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.3, random_state=6)
# X_train: 268  X_train: 623

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.fit(X2_train, y2_train)
predictions = lr.predict(X_test)
predictions2 = lr.predict(X2_test)

#Evaluate
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
accuracy_score(y_test, predictions)
accuracy_score(y2_test, predictions2)
recall_score(y_test, predictions)
recall_score(y2_test, predictions2)
precision_score(y_test, predictions)
precision_score(y2_test, predictions2)
print(pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predict not Survived', 'Predict Survived'], index=['True not Survived', 'True Survived']))
pd.DataFrame(confusion_matrix(y2_test, predictions2), columns=['Predict not Survived', 'Predict Survived'], index=['True not Survived', 'True Survived'])