import numpy as np
from matplotlib import pyplot as plt
from cv_part import split_image, process_image, show_image
import Levenshtein
from dateutil.parser import *
import datefinder
import datetime
import string
import enchant
import csv

#janky date finding: TODO: Make this work for various dates
def find_date(results, result):
	year_beg = 1990
	year_end = 2030
	date_x_threshold = 100 #TODO: Tune this for all dates
	date_y_threshold = 50

	day = 0
	month = 0
	year = 0

	((start_X, start_Y, end_X, end_Y), text) = result

	matches = datefinder.find_dates(text)

	#assume only one match TODO: make this more general
	try:
		year_found = False
		dates = []
		for match in matches:
			day = match.day
			month = match.month
			year = match.year

			#ok, we need to find the year, TODO: make this smarter
			#for now, just see if there is a date below
			for ((start_Xy, start_Yy, end_Xy, end_Yy), text) in results:

				start_diff_x = np.abs(start_Xy - start_X)
				end_diff_x = np.abs(end_Xy - end_X)
				start_diff_y = start_Yy - start_Y
				end_diff_y = end_Yy - end_Y
				total_diff_x = start_diff_x + end_diff_x
				total_diff_y = start_diff_y + end_diff_y

				if ((total_diff_y > 0) and (total_diff_y < date_y_threshold) and (total_diff_x < date_x_threshold)):
					new_text = text.split() 
					for t in new_text:
						#to = t.translate(str.maketrans(' ', ' ', string.punctuation))
						to = t.translate(str.maketrans({key: " ".format(key) for key in string.punctuation}))
						try:
							pot_year = int(to)
							if ((pot_year > year_beg) and (pot_year < year_end)):
								year = pot_year
								dates.append((day, month, year))
								year_found = True
						except:
							year = year
	except:
		print ('No Date')

	return year_found, dates

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


def get_similar_items(results, threshold=10):
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
			((start_X, start_Y, end_X, end_Y), text) = result_to_copy
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

def delete_similar_dates(results, dates, sim_items):
	print ('Deleting Similar Dates')

def get_crosshair_distance(header, date, count):
	((start_X, start_Y, end_X, end_Y), text) = header
	((start_Xd, start_Yd, end_Xd, end_Yd), textd) = date
	((start_Xc, start_Yc, end_Xc, end_Yc), textc) = count

	d1 = np.abs(start_Y - start_Yc)
	d2 = np.abs(end_Y - end_Yc)
	d3 = np.abs(start_Xd - start_Xc)
	d4 = np.abs(end_Xd - end_Xc)

	new_distance = d1+d2+d3+d4

	return new_distance


def crosshair_results(headers, dates, counts):
	total_headers = len(headers)
	total_dates = len(dates)

	results = np.zeros((total_headers, total_dates))

	for i in range(len(headers)):
		for j in range(len(dates)): 
			header = headers[i]
			date = dates[j]

			#crosshair
			((start_X, start_Y, end_X, end_Y), text) = header
			((start_Xd, start_Yd, end_Xd, end_Yd), textd) = date

			best_distance = 1e6
			best_index = 0
			#look for a result
			for k in range(len(counts)):
				count = counts[k]

				((start_Xc, start_Yc, end_Xc, end_Yc), textc) = count
				distance = get_crosshair_distance(header, date, count)

				if (distance < best_distance):
					best_distance = distance
					best_index = k

			results[i, j] = best_index

	return results

def print_results(headers, dates, dates_full, results):
	filename = 'results.csv'
	with open('results.csv', mode='w') as csv_file:
	    fieldnames = ['Field']

	    for date in dates_fuull:
	    	(day, month, year) = date
	    	d = datetime.date(year, month, day)
	    	fieldnames.append(d)

	    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	    writer.writeheader()

	    #This might not work
	    writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})
	    writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})


###** MAIN **###

#look_for = ['Total Net Sales', '2020'] #apple
#look_for = ['Revenues from franchised restaurants', '2019'] #mcds
#look_for = ['Total Sales and Revenues', '2020'] #cat

#Creating argument dictionary for the default arguments needed in the code. 
args = {"full_image":"/Users/surajmenon/Desktop/findocDocs/apple_tc_full1.png","east":"/Users/surajmenon/Desktop/findocDocs/frozen_east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}

#args['full_image']="/Users/surajmenon/Desktop/findocDocs/apple_tc_full1.png" #apple
#args['full_image']="/Users/surajmenon/Desktop/findocDocs/cat_tc_full1.png" #cat
args['full_image']="/Users/surajmenon/Desktop/findocDocs/mcds_tc_full1.png" #mcds
args['east']="/Users/surajmenon/Desktop/findocDocs/frozen_east_text_detection.pb"
args['min_confidence'] = 1e-3 #TODO: tune this
args['width'] = 320 #TODO: verify these
args['height'] = 320

#process image
# image0, results0 = process_image(image=args['image0'], args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=400, hyst_Y=15)
# (origH0, origW0) = image0.shape[:2]

# image1, results1 = process_image(image=args['image1'], args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=30, hyst_Y=5, offset_X=origW0, offset_Y=origH0)

#for printing image
#image1, results1 = process_image(image=args['image1'], args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=30, hyst_Y=5, offset_X=0, offset_Y=0)

#append results
#results = results0 + results1

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
date_results = []
dates_parsed = []
count_results = []
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
		else:
			r_image, results = process_image(False, image_to_process, args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=process_date_x, hyst_Y=process_date_y, offset_X=X_offset, offset_Y=Y_offset, remove_boxes=False)
			#r_image, results = process_image(False, image_to_process, args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=process_date_x, hyst_Y=process_date_y, offset_X=0, offset_Y=0, remove_boxes=False)

			#show_image(r_image, results)

			for ((start_X, start_Y, end_X, end_Y), text) in results:

				result = ((start_X, start_Y, end_X, end_Y), text)
				date_found, dates = find_date(results, result)

				if (date_found == True):
					for date in dates:
						(day, month, year) = date
						date_results.append(((start_X, start_Y, end_X, end_Y), text))
						dates_parsed.append(date)
				else:
					count_results.append(((start_X, start_Y, end_X, end_Y), text))

		full_results += results

#Now clean up results, remove excess headers and dates, add spellcheck
#TODO: Tune these values
header_threshold = 300
date_threshold = 150
count_threshold = 50

sim_headers = get_similar_items(header_results, threshold=header_threshold)
sim_dates = get_similar_items(date_results, threshold=date_threshold)
sim_counts = get_similar_items(count_results, threshold=count_threshold)

print ('Similar Pieces')
print (sim_headers)
print (sim_dates)
print (sim_counts)

#print (header_results)
#print_similar(header_results, sim_headers)

#now delete items based on some metric, for now just get rid of headers
trim_headers = delete_similar_headers(header_results, sim_headers)

#add something here to remove excess dates
trim_dates = delete_similar_dates(date_results, dates_parsed, sim_dates)

#delete headers and dates that don't have crosshairs

#print headers in order
print_headers(trim_headers)

print ('Current Output')
print (trim_headers)
print (date_results)
print (dates_parsed)
exit()

#Add something here for context
				
#do spellcheck, embedding check

#Now find all the crosshairs and save in an array
final_results = crosshair_results(trim_headers, date_results, count_results)

#print results
printed_results = print_results(trim_headers, date_results, dates_parsed, final_results)

#output to csv

#find coordinates
# total_coord = len(look_for)

# #probably better implemented with tuples
# X_start = np.zeros(total_coord)
# X_end = np.zeros(total_coord)
# Y_start = np.zeros(total_coord)
# Y_end =  np.zeros(total_coord)
# pieces = ['None', 'None']

# index = 0
# for look in look_for:
# 	best_dist = 1e6
# 	for ((start_X, start_Y, end_X, end_Y), text) in results:
# 		text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
# 		text = text.lower()
# 		dist = Levenshtein.distance(look, text)

# 		if (dist < best_dist):
# 			best_dist = dist
# 			X_start[index] = start_X
# 			X_end[index] = end_X
# 			Y_start[index] = start_Y
# 			Y_end[index] = end_Y
# 			pieces[index] = text

# 	index += 1

# print (pieces)

# #crosshair in, for now assume year is the column
# total_distance = 1e6
# answer = 'None'


# for ((start_X, start_Y, end_X, end_Y), text) in results:
# 	d1 = np.abs(Y_start[0] - start_Y)
# 	d2 = np.abs(Y_end[0] - end_Y)
# 	d3 = np.abs(X_start[1] - start_X)
# 	d4 = np.abs(X_end[1] - end_X)

# 	new_distance = d1+d2+d3+d4
# 	if (new_distance < total_distance):
# 		total_distance = new_distance

# 		#print ('New Best Value!')
# 		#print (text)

# 		text = "".join([x if ord(x) < 128 else "" for x in text]).strip()

# 		answer = text


# print ('We are Looking For:')
# print (look_for[0])
# print ('in')
# print (look_for[1])
# print ('And Our Answer is:')
# print (answer)
# print ('Done')