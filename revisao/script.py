import pandas as pd

df = pd.read_csv('./diabetes_clean.csv')
df = df[(df['glucose'] > 0) & (df['bmi'] > 0)]

X = df.drop('glucose', axis=1).values
y = df['glucose'].values

print(df)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

print(reg_all.score(X_test, y_test))

from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'mse: {mse}, rmse: {rmse}')

