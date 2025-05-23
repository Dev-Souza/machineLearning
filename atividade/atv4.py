import numpy as np
import pandas as pd

df = pd.read_csv('./advertising_and_sales_clean.csv')

# Create X from the radio column's values
X = df['radio'].values

# Create y from the sales column's values
y = df['sales'].values

# Reshape X
X = X.reshape(-1, 1)

# Check the shape of the features and targets
print("Shape de X:", X.shape)
print("Shape de y:", y.shape)

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()
