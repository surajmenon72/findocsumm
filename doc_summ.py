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

#split image
split_images = split_image(args['full_image'], horiz_slices=4, horiz_buffer=50, vert_slices=6, vert_buffer=50)

#process image
image0, results0 = process_image(args['image0'], args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=400, hyst_Y=15)
(origH0, origW0) = image0.shape[:2]

image1, results1 = process_image(args['image1'], args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=30, hyst_Y=5, offset_X=origW0, offset_Y=origH0)

#for printing image
#image1, results1 = process_image(args['image1'], args['east'], args['min_confidence'], args['width'], args['height'], hyst_X=30, hyst_Y=5, offset_X=0, offset_Y=0)

#append results
results = results0 + results1

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