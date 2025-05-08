import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

seeds_df = pd.read_csv('./seeds.csv', header=None)
seeds = seeds_df.to_numpy()

model = KMeans(n_clusters=4)

X_train, samples = train_test_split(seeds, test_size=0.3, random_state=42)

model.fit(X_train)
labels = model.predict(samples)
