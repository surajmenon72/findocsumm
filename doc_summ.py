import numpy as np
import pandas as pd #need openpyxl as well
from matplotlib import pyplot as plt
from cv_part import split_image, process_image, show_image
import Levenshtein
from dateutil.parser import *
import datefinder
import datetime
import string
import enchant
import csv

def get_result_distance(result1, result2):
	((start_X, start_Y, end_X, end_Y), text) = result1
	((start_Xy, start_Yy, end_Xy, end_Yy), texty) = result2

	d1 = np.abs(start_X-start_Xy)
	d2 = np.abs(end_X-end_Xy)
	d3 = np.abs(start_Y-start_Yy)
	d4 = np.abs(end_Y-end_Yy)

	total_distance = d1+d2+d3+d4
	return total_distance

def get_mid_result_distance(result1, result2):
	((start_X, start_Y, end_X, end_Y), text) = result1
	((start_Xy, start_Yy, end_Xy, end_Yy), texty) = result2

	X1 = int((end_X - start_X)/2)
	Y1 = int((end_Y - start_Y)/2)

	d1 = np.abs(X1-start_Xy)
	d2 = np.abs(X1-end_Xy)
	d3 = np.abs(Y1-start_Yy)
	d4 = np.abs(Y1-end_Yy)

	total_distance = d1+d2
	return total_distance

def find_years(results):
	year_beg = 1990
	year_end = 2030
	years = []
	year = 0
	for ((start_Xy, start_Yy, end_Xy, end_Yy), text) in results:
		new_text = text.split()
		for word in new_text:
			word_c = word.translate(str.maketrans({key: " ".format(key) for key in string.punctuation}))
			try:
				pot_year = int(word_c)
				if ((pot_year > year_beg) and (pot_year < year_end)):
					year = pot_year
					year_to_append = ((start_Xy, start_Yy, end_Xy, end_Yy), pot_year)
					years.append(year_to_append)
			except:
				year = year
	return years

def find_dates(results):
	d = enchant.Dict("en_US")
	months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
	dates = []
	for ((start_Xy, start_Yy, end_Xy, end_Yy), text) in results:
		matches = datefinder.find_dates(text)
		try:
			for match in matches:
				day = match.day
				month = match.month
				year = match.year

				#check to make sure this is real text
				words = text.split()
				current_real_words = 0
				for word in words:
					if (word in months):
						current_real_words += 1

				if (current_real_words > 0):
					date_to_append = ((start_Xy, start_Yy, end_Xy, end_Yy), text, (day, month, year))
					dates.append(date_to_append)
		except:
			print ('No Date')
	return dates


def match_years_dates(years, dates):
	date_threshold = 500 #TODO: Need to tune this
	final_dates = []
	final_dates_full = []
	for year in years:
		((start_Xy, start_Yy, end_Xy, end_Yy), year_v) = year
		date_found = False
		closest_date = 1e6
		pot_date = 0
		dt = 0
		for date in dates:
			((start_X, start_Y, end_X, end_Y), date_text, (day, month, year)) = date

			start_diff_x = np.abs(start_Xy - start_X)
			end_diff_x = np.abs(end_Xy - end_X)
			start_diff_y = np.abs(start_Yy - start_Y) #added back abs value here, allow year to be anywhere around date
			end_diff_y = np.abs(end_Yy - end_Y)
			total_diff_x = start_diff_x + end_diff_x
			total_diff_y = start_diff_y + end_diff_y
			total_diff = total_diff_x + total_diff_y

			if (total_diff < date_threshold):
				if (total_diff < closest_date):
					date_found = True
					closest_date = total_diff
					pot_date = (day, month, year_v)
					dt = date_text

		if (date_found == True):
			final_date_to_append = ((start_Xy, start_Yy, end_Xy, end_Yy), dt) # the year is the standard
			final_full_date_to_append = pot_date

			final_dates.append(final_date_to_append)
			final_dates_full.append(final_full_date_to_append)

	return final_dates, final_dates_full

def find_contexts(results):
	date_contexts = ['Months Ended', 'Months ended', 'months ended', 'Weeks Ended', 'Weeks ended', 'weeks ended']
	count_contexts = ['in millions', 'In Millions', 'In millions', 'in billions', 'In Billions', 'In billions']

	new_date_contexts = []
	new_count_contexts = []
	for result in results:
		((start_X, start_Y, end_X, end_Y), text) = result

		context_found = False
		for c in date_contexts:
			if (c in text):
				context_found = True
				s_text = text.split()
				index = 0
				for i in range(len(s_text)):
					if (s_text[i] == c[0]): #should be 'Months'
						if (i > 0):
							index = i-1

				text_to_append = s_text[index] + ' ' + s_text[index+1] + ' ' + s_text[index+2]
				context_to_append = ((start_X, start_Y, end_X, end_Y), text_to_append)
				new_date_contexts.append(context_to_append)

		if (context_found == False):
			for c in count_contexts:
				if (c in text):
					context_found = True
					text_to_append = c.lower()
					context_to_append = ((start_X, start_Y, end_X, end_Y), text_to_append)
					new_count_contexts.append(context_to_append)

	return new_date_contexts, new_count_contexts

def connect_date_contexts(dates, full_dates, date_contexts):
	date_contexts_final = []
	for date in dates:
		((start_X, start_Y, end_X, end_Y), text) = date
		best_dist = 1e6
		pot_c = 0
		for dc in date_contexts:
			((start_Xc, start_Yc, end_Xc, end_Yc), textc) = dc
			dist = get_result_distance(dc, date)
			if (dist < best_dist):
				best_dist = dist
				pot_c = dc

		date_contexts_final.append(pot_c)

	return date_contexts_final

def get_closeness(result, n_result):
	((start_X, start_Y, end_X, end_Y), text) = result
	((start_Xn, start_Yn, end_Xn, end_Yn), textn) = n_result

	#word_thresh = 5 #TODO: Tune this better, and fix this whole distance system
	frac_thresh = .3 #3 distance per 10 chars

	len1 = len(text)
	len2 = len(textn)
	min_len = np.minimum(len1, len2)

	word_thresh = frac_thresh*min_len

	dist = Levenshtein.distance(text, textn) #TODO: probably is a better way to do this
	if (dist < word_thresh):	
		startx_diff = np.abs(start_X - start_Xn)
		endx_diff = np.abs(end_X - end_Xn)
		starty_diff = np.abs(start_Y - start_Yn)
		endy_diff = np.abs(end_Y - end_Yn)

		total_diff = startx_diff+endx_diff+starty_diff+endy_diff
	else:
		#words aren't close, just let it pass
		total_diff = 1e6
	return total_diff


def get_similar_headers(results, threshold=10):
	#find similar headers, TODO: Look for better algorithm for this
	sim_results = []
	for i in range(len(results)):
		result = results[i]
		similar = []
		in_list = any(i in sublist for sublist in sim_results)
		if (in_list == False):
			similar.append(i)
			for j in range(len(results)):
				n_result = results[j]
				if (i != j):
					closeness = get_closeness(result, n_result)
					if (closeness < threshold):
							similar.append(j)

			if (len(similar) > 1):
				sim_results.append(similar)

	return sim_results

def print_similar(results, similar):
	i = 1
	for s in similar:
		print ('List')
		print (i)
		for k in range(len(s)):
			print (results[s[k]])

		i += 1

def print_headers(headers):
	ordered_headers = sorted(headers, key=lambda tup: tup[0][1])

	print ('Printing Ordered Headers')
	for oh in ordered_headers:
		((start_X, start_Y, end_X, end_Y), text) = oh
		print (text)

def sort_headers(headers):
	ordered_headers = sorted(headers, key=lambda tup: tup[0][1])

	return ordered_headers


#TODO: This func sucks, fix it to make it more efficient
def delete_similar_headers(results, sim_items):
	d = enchant.Dict("en_US")
	results_new = []
	results_final = []
	real_thresh = .75 #TODO: Tune this
	for l in sim_items:
		#keep the item that has the most real words
		most_real_words = 0
		best_i = 0
		for i in range(len(l)):
			index = l[i]
			result = results[index]
			((start_X, start_Y, end_X, end_Y), text) = result
			words = text.split()
			current_real_words = 0
			for word in words:
				if (d.check(str(word))):
					current_real_words += 1

			if (current_real_words > most_real_words):
				most_real_words = current_real_words
				best_i = i

		best_index = l[best_i]
		result_to_copy = results[best_index]
		results_new.append(result_to_copy)

	#add the non similar pieces
	for i in range(len(results)):
		found = False
		for l in sim_items:
			if (i in l):
				found = True
				break

		if (found == False):
			result_to_copy = results[i]
			results_new.append(result_to_copy)

	#now cleanup, remove punctuation, and remove those with too few real words
	for i in range(len(results_new)):
		result_to_process = results_new[i]
		((start_X, start_Y, end_X, end_Y), text) = result_to_process

		#removes punctuation
		#clean_text = text.translate(str.maketrans(' ', ' ', string.punctuation))
		clean_text = text.translate(str.maketrans({key: " ".format(key) for key in string.punctuation}))

		#check percentage of real words
		words = clean_text.split()
		num_words = len(words)
		num_real_words = 0
		for word in words:
			if (d.check(str(word))):
				num_real_words += 1

		if (num_words > 0):
			real_perc = (num_real_words/num_words)
		else:
			real_perc = 0

		if (real_perc >= real_thresh):
			result_to_append = ((start_X, start_Y, end_X, end_Y), clean_text)
			results_final.append(result_to_append)

	return results_final

def get_date_closeness(result, n_result, date_a, date_b):
	((start_X, start_Y, end_X, end_Y), text) = result
	((start_Xn, start_Yn, end_Xn, end_Yn), textn) = n_result

	#word_thresh = 5 #TODO: Tune this better, and fix this whole distance system
	frac_thresh = .5 #1 distance per 10 chars

	len1 = len(text)
	len2 = len(textn)
	min_len = np.minimum(len1, len2)

	word_thresh = frac_thresh*min_len

	dist = Levenshtein.distance(text, textn) #TODO: probably is a better way to do this
	if (dist < word_thresh):	
		startx_diff = np.abs(start_X - start_Xn)
		endx_diff = np.abs(end_X - end_Xn)
		starty_diff = np.abs(start_Y - start_Yn)
		endy_diff = np.abs(end_Y - end_Yn)

		total_diff = startx_diff+endx_diff+starty_diff+endy_diff
	else:
		#words aren't close, just let it pass
		total_diff = 1e6
	return total_diff

def get_similar_dates(results, clean_dates, threshold=10):
	sim_results = []
	for i in range(len(results)):
		result = results[i]
		similar = []
		in_list = any(i in sublist for sublist in sim_results) #TODO: Verify our vals are tuned so that this is never needed
		if (in_list == False):
			similar.append(i)
			for j in range(len(results)):
				n_result = results[j]
				if (i != j):
					closeness = get_date_closeness(result, n_result, clean_dates[i], clean_dates[j])
					if (closeness < threshold):
							also_in_list = any(j in sublist for sublist in sim_results)
							if (also_in_list == False):
								similar.append(j)

			if (len(similar) > 1):
				sim_results.append(similar)

	return sim_results

def delete_similar_dates(results, dates, sim_items):
	d = enchant.Dict("en_US")
	results_new = []
	dates_new = []
	for l in sim_items:
		#keep the item that has the most real words
		most_real_words = 0
		best_i = 0
		for i in range(len(l)):
			index = l[i]
			result = results[index]
			((start_X, start_Y, end_X, end_Y), text) = result
			words = text.split()
			current_real_words = 0
			for word in words:
				if (d.check(str(word))):
					current_real_words += 1

			if (current_real_words > most_real_words):
				most_real_words = current_real_words
				best_i = i

		best_index = l[best_i]
		result_to_copy = results[best_index]
		date_to_copy = dates[best_index]
		results_new.append(result_to_copy)
		dates_new.append(date_to_copy)

	#add the non similar pieces
	for i in range(len(results)):
		found = False
		for l in sim_items:
			if (i in l):
				found = True
				break

		if (found == False):
			result_to_copy = results[i]
			date_to_copy = dates[i]
			results_new.append(result_to_copy)
			dates_new.append(date_to_copy)

	return results_new, dates_new

def delete_false_counts(results):
	clean_results = []
	dummy = 0
	dec_place_limit = 2
	for result in results:
		((start_X, start_Y, end_X, end_Y), text) = result
		trans_dict = {}
		for key in string.punctuation:
			if (key == '.'):
				trans_dict[key] = '.'.format(key)
			#elif (key == '$'): #reject dollar signs
			#	trans_dict[key] = '$'.format(key)
			else:
				trans_dict[key] = ''.format(key)
		clean_text = text.translate(str.maketrans(trans_dict))
		#clean_text = text.translate(str.maketrans({key: "".format(key) for key in string.punctuation}))
		#clean_text = text

		try:
			count = float(clean_text)

			#try to reject high decimal points
			s_count = str(count)
			append = True
			if ('.' in s_count):
				sct = s_count.split('.')
				if (len(sct[1]) > dec_place_limit):
					append = False
			if (append == True):
				result_to_append = ((start_X, start_Y, end_X, end_Y), count)
				clean_results.append(result_to_append)
		except:
			dummy = dummy

	return clean_results

def get_crosshair_distance(header, date, count):
	((start_X, start_Y, end_X, end_Y), text) = header
	((start_Xd, start_Yd, end_Xd, end_Yd), textd) = date
	((start_Xc, start_Yc, end_Xc, end_Yc), textc) = count

	d1 = np.abs(start_Y - start_Yc)
	d2 = np.abs(end_Y - end_Yc)
	d3 = np.abs(start_Xd - start_Xc)
	d4 = np.abs(end_Xd - end_Xc)

	header_distance = d1+d2
	date_distance = d3+d4
	new_distance = d1+d2+d3+d4

	return new_distance, header_distance, date_distance


def crosshair_results(headers, dates, counts):
	total_headers = len(headers)
	total_dates = len(dates)

	results = np.zeros((total_headers, total_dates))

	header_thresh = 50 #TODO: Tune these values
	date_thresh = 300

	for i in range(len(headers)):
		for j in range(len(dates)): 
			header = headers[i]
			date = dates[j]

			#crosshair
			((start_X, start_Y, end_X, end_Y), text) = header
			((start_Xd, start_Yd, end_Xd, end_Yd), textd) = date

			best_distance = 1e6
			best_index = 0
			hd_valid = False

			#look for a result
			for k in range(len(counts)):
				count = counts[k]

				((start_Xc, start_Yc, end_Xc, end_Yc), textc) = count
				distance, hd, dd = get_crosshair_distance(header, date, count)

				if ((hd < header_thresh) and (dd < date_thresh)):
					if (distance < best_distance):
						hd_valid = True
						best_distance = distance
						best_index = k

			if (hd_valid == True):
				results[i, j] = best_index
			else:
				results[i, j] = 0

	return results

def clean_results(results):
	rows = results.shape[0]
	cols = results.shape[1]

	results_copy = results.copy()

	row_header = np.arange(rows).reshape(rows, 1) 
	col_header = (np.arange(cols+1) - 1).reshape(1, cols+1)

	results_copy = np.concatenate((row_header, results_copy), axis=1)

	results_copy = np.concatenate((col_header, results_copy), axis=0)

	rows = results_copy.shape[0]
	cols = results_copy.shape[1]

	#del cols
	col_thresh = int(rows/2)
	del_cols = []

	for j in range(cols):
		arr = results_copy[:, j]
		count = np.count_nonzero(arr)
		if (count < col_thresh):
			del_cols.append(j)

	c = 0
	for d in del_cols:
		ind = d - c
		results_copy = np.delete(results_copy, ind, axis=1)
		c += 1

	#del rows, actually just leave, and just BOLD in the printing section
	# new_cols = results_copy.shape[1]
	# row_thresh = int(new_cols-1)
	# del_rows = []

	# for i in range(rows):
	# 	arr = results_copy[i, :]
	# 	count = np.count_nonzero(arr)
	# 	if (count < row_thresh):
	# 		del_rows.append(i)

	# c = 0
	# for d in del_rows:
	# 	ind = d - c
	# 	results_copy = np.delete(results_copy, ind, axis=0)
	# 	c += 1

	return results_copy


def print_results(headers, dates, dates_full, counts, date_contexts, count_contexts, clean_results, filename):
	headers_text = []

	#select the count context, for now just pick the first one. TODO: Make this more sophis
	start_header = ''
	if (len(count_contexts) > 0):
		cc = count_contexts[0]
		((start_X, start_Y, end_X, end_Y), text) = cc
		start_header = text

	num_cols = clean_results.shape[1]-1
	col_thresh = num_cols-1 #TODO: Verify, but allow one zero
	zero_rows = []
	c = 1
	for header in headers:
		((start_X, start_Y, end_X, end_Y), text) = header
		if (c == 1): #TODO: Dumb, but make it better later
			text += '--' + start_header
		arr = clean_results[c, :]
		count = (np.count_nonzero(arr)-1) #include the column header, except for first row
		if (count < col_thresh):
			zero_rows.append(c)
			text = text.upper()
			headers_text.append(text)
		else:
			headers_text.append(text)
		c += 1

	df = pd.DataFrame({'Headers': headers_text})

	date_cols = clean_results[0, :]

	for c in range(1, num_cols+1):
		ind_dc = int(date_cols[c])
		date = dates[ind_dc]
		date_full = dates_full[ind_dc]
		c_text = ''
		print (date_contexts)
		if (len(date_contexts) > 0):
			ctxt = date_contexts[c-1]
			if (ctxt):
				((start_Xc, start_Yc, end_Xc, end_Yc), textc) = ctxt
				c_text = textc
		date_str = str(c) + '-' + str(date_full[0]) + '/' + str(date_full[1]) + '/' + str(date_full[2]) + '--' + c_text

		#fill in values
		col_arr = clean_results[:, c]
		values = []
		for i in range(1, len(col_arr)):
			if (i not in zero_rows):
				val_ind = int(col_arr[i])
				count = counts[val_ind]
				((start_Xc, start_Yc, end_Xc, end_Yc), textc) = count
				values.append(float(textc))
			else:
				values.append('--')

		#add to dateframe
		df[date_str] = values

	print (df)
	#df.to_excel(filename, sheet_name='sheet1', index=False)
	df.to_csv(filename, index=False)


###** MAIN **###

#Creating argument dictionary for the default arguments needed in the code. 
args = {"full_image":"/Users/surajmenon/Desktop/findocDocs/apple_tc_full1.png","east":"/Users/surajmenon/Desktop/findocDocs/frozen_east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}

filename = 'apple.csv'

args['full_image']="/Users/surajmenon/Desktop/findocDocs/apple_tc_full1.png" #apple
#args['full_image']="/Users/surajmenon/Desktop/findocDocs/cat_tc_full2.png" #cat
#args['full_image']="/Users/surajmenon/Desktop/findocDocs/mcds_tc_full1.png" #mcds
#args['full_image']="/Users/surajmenon/Desktop/findocDocs/gme_tc_full1.png" #gme
#args['full_image']="/Users/surajmenon/Desktop/findocDocs/adobe_tc_full1.png" #adobe
args['east']="/Users/surajmenon/Desktop/findocDocs/frozen_east_text_detection.pb"
args['min_confidence'] = 1e-3 #TODO: tune this
args['width'] = 320 #TODO: verify these
args['height'] = 320


num_horiz_slices = 4
num_vert_slices = 6
horiz_buffer = 50
vert_buffer = 50

#split image
split_images = split_image(args['full_image'], horiz_slices=num_horiz_slices, horiz_buffer=horiz_buffer, vert_slices=num_vert_slices, vert_buffer=vert_buffer)

#capture sizes, we assume all widths and heights are the same
image = split_images[0]
sub_image_width = split_images[0].shape[1]
sub_image_height = split_images[0].shape[0]

#process the sets of images
process_short_threshold = 1
header_results = []
year_results = []
dm_results = []
date_results = []
dates_parsed = []
count_results = []
context_results = []
full_results = []
process_wide_x = 400
process_wide_y = 5 #TODO: Tune y values
process_date_x = 40
process_date_y = 5

for i in range(num_vert_slices):
	for j in range(num_horiz_slices):
		index = (i*num_horiz_slices + j)
		image_to_process = split_images[index]

		#calculate offset, TODO: Verify this works even though we have the buffer
		X_offset = j*sub_image_width 
		Y_offset = i*sub_image_height

		if (j < process_short_threshold):
			r_image, results = process_image(False, image_to_process, args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=process_wide_x, hyst_Y=process_wide_y, offset_X=X_offset, offset_Y=Y_offset, remove_boxes=True)
			#r_image, results = process_image(False, image_to_process, args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=process_wide_x, hyst_Y=process_wide_y, offset_X=0, offset_Y=0, remove_boxes=True)
			header_results += results
			context_results += results
		else:
			r_image, results = process_image(False, image_to_process, args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=process_date_x, hyst_Y=process_date_y, offset_X=X_offset, offset_Y=Y_offset, remove_boxes=False)
			cr_image, cresults = process_image(False, image_to_process, args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=process_wide_x, hyst_Y=process_wide_y, offset_X=X_offset, offset_Y=Y_offset, remove_boxes=True)
			#r_image, results = process_image(False, image_to_process, args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=process_date_x, hyst_Y=process_date_y, offset_X=0, offset_Y=0, remove_boxes=False)

			#show_image(r_image, results)

			years = find_years(results)

			dm = find_dates(results)

			year_results += years
			dm_results += dm

			count_results += results
			context_results += cresults

		full_results += results

date_results_new, dates_parsed_new = match_years_dates(year_results, dm_results)
date_results += date_results_new
dates_parsed += dates_parsed_new

#find contexts
date_contexts, count_contexts = find_contexts(context_results)

#Now clean up results, remove excess headers and dates, add spellcheck
#TODO: Tune these values
header_threshold = 300
date_threshold = 150
count_threshold = 50

sim_headers = get_similar_headers(header_results, threshold=header_threshold)
sim_dates = get_similar_dates(date_results, dates_parsed, threshold=date_threshold)

#now delete items based on some metric, for now just get rid of headers
trim_headers = delete_similar_headers(header_results, sim_headers)
trim_headers = sort_headers(trim_headers)

#add something here to remove excess dates
trim_dates_r, trim_dates = delete_similar_dates(date_results, dates_parsed, sim_dates)

trim_date_contexts = connect_date_contexts(trim_dates_r, trim_dates, date_contexts)

#clean counts of non counts
trim_counts = delete_false_counts(count_results)
				
#do spellcheck, embedding check

#Now find all the crosshairs and save in an array
final_results = crosshair_results(trim_headers, trim_dates_r, trim_counts)

#delete headers and dates that don't have crosshairs, or we could do that in cross_hair results
clean_final_results = clean_results(final_results)

#print results
printed_results = print_results(trim_headers, trim_dates_r, trim_dates, trim_counts, trim_date_contexts, count_contexts, clean_final_results, filename)

print ('Done!')