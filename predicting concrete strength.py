import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


data = pd.read_csv('C:/Users/ASUS/Downloads/datalab_export_2024-10-24 18_20_39.csv')

print(data.head()) 
data= data.dropna()

target = data[['cement', 'slag','fly_ash','water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']]
feature = data['strength'] 

scaler = MinMaxScaler()
feature['age'] = scaler.fit_transform(feature[['age']])


X_train, X_test, y_train, y_test = train_test_split(feature,target, test_size= 0.3, random_state= 42)
 
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape) 

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

mse =mean_squared_error(y_test,y_pred)
print("Mean Sq error:{:.2f}", mse)

acc_score = accuracy_score(y_test,y_pred)
print("the accuracy is {:.2f}", acc_score)

r2 = r2_score(y_test, y_pred)
print("R-squared:{:.2f}", r2) 

print("Intercept (B0):", model.intercept_)
print("Coefficients (B1, B2, ..., B8):", model.coef_) 
example_input = [[300, 50, 30, 180, 5, 1000, 600, 28]]
predicted_strength = model.predict(example_input)
print("Predicted Compressive Strength:", predicted_strength[0])