import numpy as np
from matplotlib import pyplot as plt
from cv_part import split_image, process_image, show_image
import Levenshtein
from dateutil.parser import *
import datefinder
import string


#janky date finding: TODO: Make this work for various dates
def find_date(result):

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
					to = t.translate(str.maketrans('', '', string.punctuation))
					try:
						pot_year = int(to)
						if ((pot_year > year_beg) and (pot_year < year_end)):
							year = pot_year
							dates.append((day, month, year))
							year_found = True
					except:
						year = year

	return year_found, dates


###** MAIN **###

#look_for = ['Total Net Sales', '2020'] #apple
#look_for = ['Revenues from franchised restaurants', '2019'] #mcds
look_for = ['Total Sales and Revenues', '2020'] #cat

#Creating argument dictionary for the default arguments needed in the code. 
args = {"full_image":"/Users/surajmenon/Desktop/findocDocs/apple_tc_full1.png","image0":"/Users/surajmenon/Desktop/findocDocs/apple_test1.png", "image1":"/Users/surajmenon/Desktop/findocDocs/apple_test1.png", "east":"/Users/surajmenon/Desktop/findocDocs/frozen_east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}

args['full_image']="/Users/surajmenon/Desktop/findocDocs/apple_tc_full1.png"
args['image0']="/Users/surajmenon/Desktop/findocDocs/cat_tc1.png"
args['image1']="/Users/surajmenon/Desktop/findocDocs/cat_tc2.png"
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
count_results = []
full_results = []
process_wide_x = 400
process_wide_y = 4
process_date_x = 40
process_date_y = 4

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
				date_found, dates = find_date(result)

				if (date_found == True):
					for date in dates:
						(day, month, year) = date
						date_results.append(((start_X, start_Y, end_X, end_Y), text, (day, month, year)))
				else:
					count_results.append(((start_X, start_Y, end_X, end_Y), text))

		full_results += results

#Now clean up results, remove excess headers and dates, add spellcheck



#show image
#show_image(image0, results0)
#exit()

#do spellcheck, embedding check, join texts that are close horizontally

#find coordinates
total_coord = len(look_for)

#probably better implemented with tuples
X_start = np.zeros(total_coord)
X_end = np.zeros(total_coord)
Y_start = np.zeros(total_coord)
Y_end =  np.zeros(total_coord)
pieces = ['None', 'None']

index = 0
for look in look_for:
	best_dist = 1e6
	for ((start_X, start_Y, end_X, end_Y), text) in results:
		text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
		text = text.lower()
		dist = Levenshtein.distance(look, text)

		if (dist < best_dist):
			best_dist = dist
			X_start[index] = start_X
			X_end[index] = end_X
			Y_start[index] = start_Y
			Y_end[index] = end_Y
			pieces[index] = text

	index += 1

print (pieces)

#crosshair in, for now assume year is the column
total_distance = 1e6
answer = 'None'


for ((start_X, start_Y, end_X, end_Y), text) in results:
	d1 = np.abs(Y_start[0] - start_Y)
	d2 = np.abs(Y_end[0] - end_Y)
	d3 = np.abs(X_start[1] - start_X)
	d4 = np.abs(X_end[1] - end_X)

	new_distance = d1+d2+d3+d4
	if (new_distance < total_distance):
		total_distance = new_distance

		#print ('New Best Value!')
		#print (text)

		text = "".join([x if ord(x) < 128 else "" for x in text]).strip()

		answer = text


print ('We are Looking For:')
print (look_for[0])
print ('in')
print (look_for[1])
print ('And Our Answer is:')
print (answer)
print ('Done')