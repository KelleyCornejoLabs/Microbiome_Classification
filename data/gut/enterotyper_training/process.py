import pandas as pd
import numpy as np

# df = pd.read_csv("processed.csv")
# enteros = pd.read_csv("enterotype_data_for_Alex.csv")

# Given dataframe with just data and sampleIDs, format for enterotyper
def for_entero(x, sampleID_col):
	x = x.T
	x.columns = x.iloc[0]
	x = x.iloc[1:]
	x = x.reset_index(names="g__genus")
	cols = list(x.columns)
	cols.remove("g__genus")
	cols.remove(sampleID_col)
	x[cols] = x[cols].div(x[cols].sum(axis=0),axis=1)
	return x	
