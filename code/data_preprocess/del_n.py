import os
import pandas as pd


df = pd.read_csv("../data/aug_split/train.csv")
df['text'] = df['text'].replace('\n', ' ', regex=False)
df.to_csv("../data/aug_split/train_n.csv", index=False)