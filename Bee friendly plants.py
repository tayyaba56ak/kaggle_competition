
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


data = pd.read_csv('C:/Users/ASUS/Downloads/datalab_export_2024-10-24 15_36_13.csv') 
print(data.head()) 

#data.info() 
missing_value = data.isnull().sum().any() 
data.drop_duplicates()

data = data.dropna()
print("shape of data after handling missing vlaues" , data.shape) 
data.head() 

plt.figure(figsize=(12,6))
sns.histplot(data['bee number'])
plt.title("distribution of bees")
plt.xlabel("no. of bees")
plt.ylabel("frequency")
plt.show() 

# Identifying correlations between variables
plt.figure(figsize=(12,6)) 
correlation_matix = data.corr() 
sns.heatmap(correlation_matix,annot= True,cmap='coolwarm')
plt.title(" Correlation Matrix")
plt.show() 


target = 'bees_num' # selecting target
features = data.drop(columns=[target]) # Select the feature variables, excluding target variable)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, data[target], test_size= 0.3 , random_state= 42)
print("Training set shape (features):", x_train.shape)
print("Training set shape (target):", y_train.shape)
print("Testing set shape (features):", x_test.shape)
print("Testing set shape (target):", y_test.shape)


model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict( x_test)

acc_Score = accuracy_score(y_test,y_pred)
r_score = r2_score(y_test,y_pred)

print("Accuracy score:{:.2f}", acc_Score)
print("R-squared (R2) score:{:.2f}", r_score) 


# Display the shapes of the training and testing sets
plt.figure(figsize=(12,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


#Create a pipeline with scaling and linear regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train the pipeline on the training data
pipeline.fit(x_train, y_train)

# Make predictions on the testing data
y_pred_scaled = pipeline.predict(X_test)

# Evaluate the model's performance
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print("Mean Squared Error (MSE) after scaling:", mse_scaled)
print("R-squared (R2) Score after scaling:", r2_scaled)

# Plotting the actual vs predicted values after scaling
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_scaled, alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values after Scaling')
plt.show()









