from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')


# Lê o CSV
churn_df = pd.read_csv('./telecom_churn_clean.csv')
y = churn_df["churn"].values
X = churn_df.drop('churn', axis=1).values

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cria e treina o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Avalia com os dados de teste
accuracy = knn.score(X_test, y_test)
print(f"Acurácia: {accuracy:.2f}")