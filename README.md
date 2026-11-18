## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="233" height="237" alt="image" src="https://github.com/user-attachments/assets/8892a565-75cb-4af1-9381-3616e41c0c4b" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="137" height="133" alt="image" src="https://github.com/user-attachments/assets/faa7a11c-56d4-447f-8f06-ea3bd0b12907" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="245" height="237" alt="image" src="https://github.com/user-attachments/assets/7470cd0c-0729-4f0e-959e-5d5f40031df5" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="223" height="242" alt="image" src="https://github.com/user-attachments/assets/f237a91f-3884-4dd6-8021-f6d7a170c621" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="327" height="248" alt="image" src="https://github.com/user-attachments/assets/dec77d69-090e-4c60-8767-2bb784d120af" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="497" height="240" alt="image" src="https://github.com/user-attachments/assets/44623935-5f60-4183-b81d-a3ecf49edee6" />

```
pip install --upgrade category_encoders
```
<img width="862" height="193" alt="image" src="https://github.com/user-attachments/assets/05fe8675-cc54-4e47-a92e-98c77efdf355" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb=df.copy()
dfb
```
<img width="383" height="253" alt="image" src="https://github.com/user-attachments/assets/2ac5f239-bbb8-45b4-bad4-045572802af1" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
<img width="446" height="242" alt="image" src="https://github.com/user-attachments/assets/cd50bfe6-a9fc-47fd-9c41-7509d828848d" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="587" height="277" alt="image" src="https://github.com/user-attachments/assets/4c4f19bc-8438-4e39-97cb-ded7fed4bdf9" />

```
df.skew()
```
<img width="245" height="142" alt="image" src="https://github.com/user-attachments/assets/49baf1f3-0e72-4a16-b67f-63ca34a2831e" />

```
np.log(df["Highly Positive Skew"])
```
<img width="267" height="301" alt="image" src="https://github.com/user-attachments/assets/64b42356-bbfa-48f2-8354-fb41f638cbab" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="222" height="306" alt="image" src="https://github.com/user-attachments/assets/202c011c-99cb-40cd-9f3e-8c80b5a2619c" />

```
np.square(df["Highly Positive Skew"])
```
<img width="248" height="300" alt="image" src="https://github.com/user-attachments/assets/c05fdf1e-f614-405f-a36c-44ffe123b18b" />

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df.skew()
```
<img width="312" height="202" alt="image" src="https://github.com/user-attachments/assets/a49637da-45d4-401f-aa21-80f92d417f45" />


```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
<img width="368" height="177" alt="image" src="https://github.com/user-attachments/assets/f5b600da-34ae-4740-a624-e01b6408d73e" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="341" height="205" alt="image" src="https://github.com/user-attachments/assets/0abfd2c0-4881-4332-89cd-5bbed2525776" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
<img width="457" height="302" alt="image" src="https://github.com/user-attachments/assets/04fb8118-00b3-4e23-b285-e38d346833fd" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
```
<img width="476" height="312" alt="image" src="https://github.com/user-attachments/assets/033f200e-bd35-4dc4-bc0c-48b2b4039dba" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
```
<img width="527" height="588" alt="image" src="https://github.com/user-attachments/assets/370ba8ec-efb3-4543-804b-b62d0bc4e217" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="478" height="302" alt="image" src="https://github.com/user-attachments/assets/d56bb09a-65e1-4427-a3b0-25913bebf55f" />

## Summary:
In this experiment, I performed Feature Encoding and Feature Transformation techniques using three datasets: Encoding Data.csv, data.csv and Data_to_Transform.csv.First, I used Feature Encoding to convert categorical data into numerical values. I applied Ordinal Encoding to the ord_2 column because its categories (Hot, Warm, Cold) have an order. Label Encoding was also used to assign numerical labels to categories. One-Hot Encoding was applied to the nom_0 column because it contains nominal data without any specific order. Binary Encoding was used to represent categorical values using fewer columns, while Target Encoding converted categories into numerical values based on their relationship with the target variable.Next, I performed Feature Transformation to reduce skewness and improve the distribution of numerical data. I used Log and Square Root transformations to reduce high positive skewness. Square Transformation was used to observe changes in the data distribution. Box-Cox Transformation was applied to highly positively skewed data because it automatically selects a suitable power value. Yeo-Johnson Transformation was used for moderately and highly negatively skewed data because it supports negative values. I also applied Quantile Transformation to convert the data into an approximately normal distribution. Finally, Q-Q plots were used to visually compare the data distribution with a normal distribution.I selected each technique based on the type and distribution of the data. I did not use the same method for every column because different datasets require different approaches. For example, One-Hot Encoding is suitable for unordered categories, Ordinal Encoding is suitable for ordered categories, Box-Cox requires positive values, and Yeo-Johnson can handle positive, zero and negative values.

# RESULT:

 Thus the given data, Feature Encoding, Transformation process and save the data to a file was successfully executed.

       
