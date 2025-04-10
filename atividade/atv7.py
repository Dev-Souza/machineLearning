import pandas as pd

dados = [
    [13000.0, 9237.76, 2409.57, 46677.90],
    [41000.0, 15886.45, 2913.41, 150177.83]
]

sales_df = pd.DataFrame(dados, columns=['TV', 'RADIO', 'SOCIAL_MEDIA', 'SALES'])

X = sales_df.drop('SALES', axis=1).values
y = sales_df['SALES'].values 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg_all = LinearRegression()

# Fit the model to the data
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

from sklearn.metrics import mean_squared_error
import numpy as np

r_squared= mean_squared_error(y_test, y_pred)
rmse = np.sqrt(r_squared)
print(f'mse: {r_squared}, rmse: {rmse}')