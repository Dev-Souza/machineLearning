import pandas as pd

churn_df = pd.read_csv('telecom_churn_clean.csv')
# print(churn_df.head(5))

from sklearn.neighbors import KNeighborsClassifier
X = churn_df[["total_day_charge", "total_eve_charge"]].values
y = churn_df["churn"].values
# print(X.shape, y.shape)

# print(X[0], y[0])
# print(X[1], y[1])

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)

import numpy as np

X_new = np.array([[56.8, 17.5],  # primeira observação
                  [24.4, 24.1],  # segunda  observação
                  [50.1, 10.9]]) # terceira observação
print(X_new.shape)

predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))