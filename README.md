# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.ARUVI
RegisterNumber:  212222230014
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1) # removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test , y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1= classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]) 
```



## Output:
### 1.PLACEMENT DATA


![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/8e7a996c-000e-4c33-a1f2-9d1adc121b17)

### 2.SALARY DATA

![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/89fb980c-5ea1-4dcf-b529-b95d4e990e83)

### 3.CHECKING THE NULL FUNCTION()

![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/728923bc-6a99-4032-bf2e-c0a47697b830)

### 4.DATA DUPLICATE

![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/4deb29cb-6eec-4eeb-a8ca-8afdc1f47a25)

### 5.PRINT DATA

![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/7c0fa3ad-f8d0-4e9c-89b3-ae51e6b3c892)

### 6. DATA STATUS

![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/d31a470c-936c-469e-84f9-5723b18a461d)

### 7. y_prediction array

![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/17ff72f3-90d2-44de-bb68-728f4693861c)

### 8.Classification Report

![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/7841b2e0-e168-4321-910c-a8bdaaf07688)

### 9. PREDICTION OF LR

![image](https://github.com/Anandanaruvi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120443233/c28782f0-9435-4ef0-b6b4-53507bcf1f26)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
