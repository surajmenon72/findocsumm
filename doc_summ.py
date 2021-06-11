import numpy as np
from matplotlib import pyplot as plt
from cv_part import split_image, process_image, show_image
import Levenshtein

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
args['min_confidence'] = 1e-3
args['width'] = 160
args['height'] = 160

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

#split image
split_images = split_image(args['full_image'], horiz_slices=num_horiz_slices, horiz_buffer=50, vert_slices=num_vert_slices, vert_buffer=50)

#capture sizes
sub_image_width = split_images[0].shape[1]
sub_image_height = split_images[0].shape[0]

#process the sets of images
process_short_threshold = 1
header_results = []
date_counts_results = []
full_results = []
process_wide_x = 400
process_wide_y = 15
process_short_x = 30
process_short_y = 5

for i in range(num_vert_slices):
	for j in range(num_horiz_slices):
		index = (i*num_horiz_slices + j)
		image_to_process = split_images[index]

		#calculate offset
		X_offset = j*sub_image_width
		Y_offset = i*sub_image_height

		#dumb, sending image0 when it does nothing: TODO: FIX
		if (j > process_short_threshold):
			r_image, results = process_image(args['image0'], args['east'], args['min_confidence'], args['width'], args['height'], image_real=image_to_process, hyst_X=process_wide_x, hyst_Y=process_wide_y, offset_X=X_offset, offset_Y=Y_offset)
			header_results += results
		else:
			r_image, results = process_image(args['image0'], args['east'], args['min_confidence'], args['width'], args['height'], image_real=image_to_process, hyst_X=process_short_x, hyst_Y=process_short_y, offset_X=X_offset, offset_Y=Y_offset)
			date_counts_results += results

		full_results += results

#Now clean up results, remove punctuation and duplicates

#Select headers and mark dates if they are there

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