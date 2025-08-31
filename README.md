# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:

Program to implement the linear regression using gradient descent.
Developed by: MOHAMMED PARVEZ S
RegisterNumber: 212223040113
```PYTHON
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros (X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
print(data.head())

```

<img width="622" height="136" alt="Screenshot 2025-08-31 150405" src="https://github.com/user-attachments/assets/fc8f32db-7051-4f91-bd52-6cb44897856b" />

```PYTHON
X = (data.iloc[1:, :-2].values) 
print (X)
```

<img width="575" height="713" alt="Screenshot 2025-08-31 150516" src="https://github.com/user-attachments/assets/32428d90-ed6d-4fa9-805d-48f8b119f105" />

```PYTHON
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)

```

<img width="373" height="732" alt="Screenshot 2025-08-31 150634" src="https://github.com/user-attachments/assets/a5d983b2-10a5-4cb2-b581-dcb2ee8ddcab" />

```PYTHON
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)

```

<img width="468" height="714" alt="Screenshot 2025-08-31 150940" src="https://github.com/user-attachments/assets/4ac5cda6-0804-4d91-a040-27615af13ee8" />

```PYTHON
print(Y1_Scaled)

```

<img width="328" height="720" alt="Screenshot 2025-08-31 151007" src="https://github.com/user-attachments/assets/aea909ce-639b-41c8-88bb-6d4495500e4a" />

```PYTHON
print('Name: MOHAMMED PARVEZ S'    )
print('Register No.:212223040113'    )
theta = linear_regression (X1_Scaled, Y1_Scaled)
new_data = np.array ([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:

<img width="1920" height="1080" alt="Screenshot 2025-08-31 151118" src="https://github.com/user-attachments/assets/e3aacd32-e219-4595-8bc0-40fe07daf160" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
