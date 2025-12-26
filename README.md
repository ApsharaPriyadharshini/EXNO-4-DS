# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("bmi.csv")

df.head()

<img width="1236" height="212" alt="image" src="https://github.com/user-attachments/assets/1981ff66-aff9-4a7a-ad9d-f5e7bf90da22" />

df.dropna()
<img width="1247" height="442" alt="image" src="https://github.com/user-attachments/assets/52a6cd32-d672-4bc3-8a07-e5544f91f029" />

max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

<img width="1258" height="177" alt="image" src="https://github.com/user-attachments/assets/81ac07a6-71b6-4daf-8d9f-dc4402131751" />

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

<img width="1248" height="379" alt="image" src="https://github.com/user-attachments/assets/914fbfa2-1aa4-4c6a-8c0b-411573d8a7e6" />

df3=pd.read_csv("bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3

<img width="1244" height="450" alt="image" src="https://github.com/user-attachments/assets/e718382b-de5b-41b6-8290-4b19e71c1df0" />

df4=pd.read_csv("bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()

<img width="1258" height="233" alt="image" src="https://github.com/user-attachments/assets/100657b6-1701-4be4-8144-d63afef79933" />

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear.model import LinearRegression 
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV,LassoCV,Ridge,Laao
from sklearn.feature_selection import SelectKbest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2

df=pd.read_csv("titanic_dataset.csv")
df.columns

<img width="1245" height="83" alt="image" src="https://github.com/user-attachments/assets/8f8f132b-1253-4ae1-b18e-6287c3f28df0" />

df.shape

<img width="1237" height="41" alt="image" src="https://github.com/user-attachments/assets/f7046d69-c958-4e3f-a6b7-12354c936b4b" />

x = df.drop("Survived",1)
y = df['Survived']


<img width="1258" height="86" alt="image" src="https://github.com/user-attachments/assets/52461175-101a-4219-b0cc-5b01105ccfa6" />

df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df1.columns


<img width="1253" height="44" alt="image" src="https://github.com/user-attachments/assets/8e00e8cc-6af8-46ce-8685-704c78bd1a9f" />

df1['Age'].isnull().sum()

<img width="1254" height="34" alt="image" src="https://github.com/user-attachments/assets/6222a435-e8a5-48c2-a841-3b6e4fb02a1b" />

df1['Age'].fillna(method='ffill')

<img width="1244" height="273" alt="image" src="https://github.com/user-attachments/assets/4cba15ef-f77a-4d92-912e-17d8eccf66f5" />

df1['Age']=df1['Age'].fillna(method='ffill')
df1['Age'].isnull().sum()

<img width="1244" height="40" alt="image" src="https://github.com/user-attachments/assets/d0b9b335-d592-42fe-850f-a6717e7fe621" />

df1.columns

<img width="1243" height="39" alt="image" src="https://github.com/user-attachments/assets/34b98ff7-3713-4277-b3d1-0c1d0c7dd328" />

cols=df1.columns.tolist()
cols[-1],cols[1]=cols[1],cols[-1]
df1=df1[cols]

df1.columns


<img width="1250" height="46" alt="image" src="https://github.com/user-attachments/assets/3afc300a-7533-4c5c-85cd-4cac5f188f0a" />

x=df1.iloc[:,0:6]
y=df1.iloc[:,6]

x.columns


<img width="1250" height="40" alt="image" src="https://github.com/user-attachments/assets/739f765e-e517-422b-8800-9009b196e0db" />

y=y.to_frame()
y.columns

<img width="1248" height="52" alt="image" src="https://github.com/user-attachments/assets/dee6ca85-a217-4012-a7f7-e389473130fd" />

import pandas as pd
from sklearn.feature_selection import selectkBest
from sklearn.feature_selection import chi2

data=pd.read_csv("titanic_dataset.csv")
data=data.dropna()
x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
X

<img width="1270" height="448" alt="image" src="https://github.com/user-attachments/assets/a76d3a61-8792-4e2f-aee3-687ee940d749" />

data

<img width="1248" height="454" alt="image" src="https://github.com/user-attachments/assets/83603f4e-e077-4ec9-b682-8d0f7fc34c24" />

x.info()


<img width="1245" height="352" alt="image" src="https://github.com/user-attachments/assets/dbbea4b0-35d3-4d4e-b4e3-6e33acb40561" />

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("titanic_dataset.csv")

df.columns

<img width="1234" height="83" alt="image" src="https://github.com/user-attachments/assets/eb846439-4dd3-4dcc-a9c0-daee26705284" />

df

<img width="1248" height="445" alt="image" src="https://github.com/user-attachments/assets/534b19d2-facf-45bc-9522-c627299a4275" />

df=df.dropna()
df.isnull().sum()


<img width="1249" height="290" alt="image" src="https://github.com/user-attachments/assets/9db840ed-40e7-4f1a-bb7e-0027ade05b82" />

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')

tips.head()

<img width="1252" height="229" alt="image" src="https://github.com/user-attachments/assets/dd9d3a8c-b745-428f-971a-9be14a6d316c" />

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)


<img width="1247" height="107" alt="image" src="https://github.com/user-attachments/assets/0aa83158-60e9-4f6d-9742-646d479529e6" />



























# RESULT:
       Thus,the feature scaling and feature selection methods are executed successfully and the output is verified
       
