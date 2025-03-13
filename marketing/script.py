import pandas as pd
import numpy as np

df = pd.read_csv('./marketing.csv', delimiter=',')

# a) Imprima as primeiras 5 linhas do DatFrame marketing
print(df.head(5))

# b) Imprima um resumo estatístico de todas as colunas do DataFrame marketing
print(df.describe(include='all'))

# c) Imprima tipos de dados das colunas e a quantidade de valores não-nulos por coluna
print(df.info())

print(df['converted'].head(5))

print(df['converted'].dtype)

df['converted'] = df['converted'].astype('bool')

print(df['converted'].dtype)

# Sobre numpy
df['is_house_ads'] = np.where(df['marketing_channel'] == 'House Ads', True, False)

print(df[['marketing_channel', 'is_house_ads']])

channel_dict = {
    'House Ads': 1,
    'Instagram': 2,
    'Facebook': 3,
    'Email': 4,
    'Push': 5,
    '': 0
}

# Set na values to default value
print(df['marketing_channel'].fillna('').map(channel_dict))
