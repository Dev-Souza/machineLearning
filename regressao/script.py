import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

diabetes_df = pd.read_csv("diabetes_clean.csv")
# print(diabetes_df.sort_values(by='bmi'))
# print(diabetes_df.head(5))

# Limpeza dos dadoa que são inválidos
diabetes_df = diabetes_df[
    (diabetes_df['bmi'] > 0) &
    (diabetes_df['glucose'] > 0)
]

X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_bmi = X[:, 4]
# print(X_bmi) MOSTRANDO OS DADOS DA COLUNA BMI
# print(y.shape, X_bmi.shape)

# -1 é equivalente a última linha, e 1 afirma que temos apenas uma única coluna.  
X_bmi = X_bmi.reshape(-1, 1)  # cria uma matriz 2D de uma única coluna.
# print(y.shape, X_bmi.shape)

# plt.scatter(X_bmi, y)
# plt.xlabel("Body Mass Index")
# plt.ylabel("Blood Glucose (mg/dl)")

# plt.show()

reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()