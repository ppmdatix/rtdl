import pandas as pd
import csv
import sys
from one_hot_encode import one_hot_encode


directory = "examples/data/" + sys.argv[1] + "/"
filename = sys.argv[2]
category_features_file = sys.argv[3]

if len(sys.argv) > 4:
    target = sys.argv[4]
else:
    target = "target"

with open(directory + category_features_file, newline='') as f:
    reader = csv.reader(f)
    cat_feat = list(reader)[0]

df = pd.read_csv(directory + filename)
df["target"] = df[target]
df = df.drop(target, axis=1)
oldNames = df.columns
output = df.target.values
labels = set(output)

for c in df.columns:
    if (not c in cat_feat) and (c != "target"):
        df = df.drop(c, axis=1)

for col in df.columns:
    if col != "target":
        df = one_hot_encode(df, col)
        df = df.drop(col, axis=1)

df["target"] = df["target"].apply(lambda x: str(x))

df.to_csv(directory + "training_processed.csv", index=False)


