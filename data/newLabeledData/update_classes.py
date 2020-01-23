import os
import pandas as pd
import numpy as np
os.chdir("/home/gozde/currentSets/newLabeledData")

df = pd.DataFrame([])
for f in os.listdir("."):
	if os.path.splitext(f)[1] == ".txt":
		df_tmp = pd.read_csv(open(f, "r"), sep=" ", header=None)
		df_tmp.columns = ["label", "x1", "y1", "x2", "y2"]
		df_tmp["file"] = f
		df = df.append(df_tmp)

print(df)

conversions_df = pd.read_csv(open("label_mapping.csv","r"),sep="\t")
#print(conversions_df)


conversion_dict = {}
for i,row in conversions_df.iterrows():
	#print(row)
	conversion_dict[row["previous"]] = row["current"]


print(conversion_dict)

#for i, row in df.iterrows():
#	print(row['label'], conversion_dict[row['label']])
#	df.loc[i,'label'] = conversion_dict[row['label']]

df["label"] = df["label"].map(lambda x: conversion_dict[x])

print(df)

for filename in np.unique(df["file"].values):
	df_tmp = df[df["file"] == filename]
	print(filename)
	print(df_tmp)
	#df_tmp.to_csv(open("new"+filename, "w"), sep=" ",columns=["label", "x1", "y1", "x2", "y2"],header=False, index=False)
	df_tmp.to_csv(open( '{}'.format(filename), "w"), sep=" ",float_format = '%f', columns=["label", "x1", "y1", "x2", "y2"],header=False, index=False)

