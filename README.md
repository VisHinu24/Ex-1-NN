<H3>Vishinu H</H3>
<H3>212223220124</H3>
<H3>EX-NO : 1</H3>
<H1> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

data.info()

print("Missing Values: \n ",data.isnull().sum())

print("Duplicate values:\n ")
print(data.duplicated())

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print("Normalized data \n" , df1)

X = data.drop('Exited', axis=1)  
y = data['Exited'] 

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training data")
print(X_train)
print(y_train)

print("Testing data")
print(X_test)
print(y_test)
print("Length of X_test: ", len(X_test))


```

## OUTPUT:

## data.info()
![image](https://github.com/user-attachments/assets/6e6df7e2-8890-40c2-a10e-04c26051c330)

## Missing values
![image](https://github.com/user-attachments/assets/b524e626-bede-4d2d-8c0b-8ac6794ed807)

## Duplicate values
![image](https://github.com/user-attachments/assets/a5ca3da8-fd56-47e6-8a34-a0fa6cb7a769)

## Normalized dataset
![image](https://github.com/user-attachments/assets/700d9f65-8c3f-4821-b1f2-b4c2c47e9239)

## Training values of X and Y
![image](https://github.com/user-attachments/assets/35796089-ba57-4b96-88a4-5c47930269e2)

## Testing values of X and Y
![image](https://github.com/user-attachments/assets/d659ebb6-c6c6-4d01-8db6-5049a0be4fe7)




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


