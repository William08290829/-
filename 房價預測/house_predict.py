import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Price.csv")

#資料觀察
df.head()
df.describe().T     #std 標準差
df.columns
df.info()     #1460個資料  81個欄位 

# sns.distplot(df['SalePrice'])
# sns.jointplot(df['GrLivArea'],df['SalePrice'])     #GrLivArea(Ground Living Area)房屋扣除地下室後的日常生活空間大小
# sns.jointplot(df['TotalBsmtSF'],df['SalePrice'])    #Total square feet of basement area
# sns.jointplot(df['EnclosedPorch'],df['SalePrice'])     #封閉式門廊面積
# sns.jointplot(df['EnclosedPorch'],df['SalePrice'])

#drop
df.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True)


#相關係數
corr = df.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]  #np.argsort()表示返回其排序的索引


# #各自的相關係數
# corr = data.corr()["SalePrice"]
# corr[np.argsort(corr, axis=0)[::-1]]
## 各自與SalePrice的關係圖
# sns.jointplot(df_data['GrLivArea'],df_data['SalePrice'])     #GrLivArea(Ground Living Area)房屋扣除地下室後的日常生活空間大小
# sns.jointplot(df_data['TotalBsmtSF'],df_data['SalePrice'])    #TotalBsmtSF(Total square feet of basement area) 地下室空間
# sns.jointplot(df_data['EnclosedPorch'],df_data['SalePrice'])     #EnclosedPorch (Enclosed porch area in square feet) 封閉式門廊面積
# sns.jointplot(df_data['1stFlrSF'],df_data['SalePrice'])         #1stFlrSF (First Floor square feet)一樓大小
# sns.jointplot(df_data['FullBath'],df_data['SalePrice'])     #FullBath(Full bathrooms above grade)全浴室以上等級
# sns.jointplot(df_data['LowQualFinSF'],df_data['SalePrice'])     #LowQualFinSF(Low quality finished square feet (all floors))低質量成品
# sns.jointplot(df_data['OverallQual'],df_data['SalePrice'])      #OverallQual整體材料和完工質量
# sns.jointplot(df_data['GarageCars'],df_data['SalePrice'])       #GarageCars車庫容許車輛
# sns.jointplot(df_data['HouseStyle'],df_data['SalePrice'])       #HouseStyle房子類型
# sns.jointplot(df_data['Id'],df_data['SalePrice'])               #ID編號
# sns.jointplot(df_data['YearBuilt'],df_data['SalePrice'])        #YearBuilt建造年份
# sns.jointplot(df_data['RoofStyle'],df_data['SalePrice'])  


#GrLivArea(Ground Living Area)房屋扣除地下室後的日常生活空間大小
#TotalBsmtSF(Total square feet of basement area) 地下室空間
#EnclosedPorch (Enclosed porch area in square feet) 封閉式門廊面積
#1stFlrSF (First Floor square feet)一樓大小
#FullBath(Full bathrooms above grade)全浴室以上等級
#LowQualFinSF(Low quality finished square feet (all floors))低質量成品
#OverallQual整體材料和完工質量
#GarageCars車庫容許車輛
#HouseStyle房子類型
#ID編號
#YearBuilt建造年份
