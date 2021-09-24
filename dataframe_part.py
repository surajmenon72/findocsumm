import numpy as np
import pandas as pd #need openpyxl as well
import string
import copy

db_path = './db/'

def save_or_append_dataframe(df, frame_name=None):
	#assume csv format
	save_path = db_path + frame_name + '.csv'
	try:
		prev_df = pd.read_csv(save_path)
	except:
		print ('No Previous Frame, starting a new one')
		df.to_csv(save_path)
		return

	#append prev and save over
	if (prev_df != None):
		full_df = pd.concat([prev_df, df], axis=1, join='inner')
		full_df.to_csv(save_path)

	return full_df
