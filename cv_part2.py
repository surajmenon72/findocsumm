import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import copy

#predictions function
def predictions(prob_score, geo, min_confidence):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < min_confidence:
				#print (scoresData[i])
				#print ('Low Confidence!')
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred bbox
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)

def connect_horizontal_boxes(boxes, x_threshold=30, y_threshold=30):
	boxes_copy = boxes.copy()
	box_it = sorted(boxes_copy, key=lambda tup: tup[0])

	done = False
	while (done == False):
		merger = (1e6, 1e6)
		box_to_merge = (0, 0, 0, 0)
		found = False
		i = 0
		for box in box_it:
			(start_X, start_Y, end_X, end_Y) = box
			j = 0
			for new_box in box_it:
				if (i < j):
					(start_Xn, start_Yn, end_Xn, end_Yn) = new_box
					startYdiff = np.abs(start_Yn - start_Y)
					endYdiff = np.abs(end_Yn - end_Y)
					Ydiff = startYdiff + endYdiff
					if (Ydiff < y_threshold):
						Xdiff = np.abs(start_Xn - end_X) 
						if ((start_Xn <= end_X) or (Xdiff < x_threshold)):
							merger = (i, j)
							sY = np.minimum(start_Y, start_Yn)
							eY = np.maximum(end_Y, end_Yn)
							found = True

							if (start_Xn <= end_X):
								eX = np.maximum(end_X, end_Xn)
								box_to_merge = (start_X, sY, eX, eY)
							else:
								box_to_merge = (start_X, sY, end_Xn, eY)
							break
				j += 1
			if (found == True):
				break
			i += 1

		#delete merger, and add new box, assume i before j
		if (found == True):
			box_change = copy.deepcopy(box_it)
			box_change.pop(merger[0])
			box_change.pop(merger[1]-1)
			box_change.append(box_to_merge)
			box_change = sorted(box_change, key=lambda tup: tup[0])
			box_it = copy.deepcopy(box_change)
		else:
			done = True

	return box_it


# def remove_nearby_boxes(boxes, threshold=10):

# 	boxes_copy = boxes.copy()

# 	#remove boxes that are close in the y-direction
# 	index_1 = 0
# 	index_remove = []
# 	for (startX, startY, endX, endY) in boxes_copy:
# 		if (index_1 not in index_remove):
# 			index_2 = 0
# 			for (startX_c, startY_c, endX_c, endY_c) in boxes_copy:
# 				#TODO: Consider something smarter here
# 				if ((startX != startX_c) and (endX != endX_c)):
# 					diff_start = np.abs(startY_c - startY)
# 					diff_end = np.abs(endY_c - endY)

# 					if ((diff_start+diff_end) < threshold):
# 						if (index_2 not in index_remove):
# 							index_remove.append(index_2)
# 				index_2 += 1
# 		index_1 += 1

# 	#remove index from copy
# 	removed = 0
# 	index_remove.sort()
# 	for i in index_remove:
# 		i_remove = i - removed
# 		boxes_copy = np.delete(boxes_copy, i_remove, axis=0)
# 		removed += 1

# 	return boxes_copy

######** MAIN **#########

#takes in a full page image and returns a split image according to specifications
# def split_image(input_image, horiz_slices=2, horiz_buffer=5, vert_slices=2, vert_buffer=5):

# 	image = cv2.imread(input_image)

# 	orig = image.copy()
# 	(origH, origW) = image.shape[:2]

# 	#lets assume for now these divisions don't need to exactly divide it due to our buffer
# 	slice_width = int(origW/horiz_slices)
# 	slice_height = int(origH/vert_slices)

# 	output_images = []

# 	for i in range(vert_slices):
# 		for j in range(horiz_slices):
# 			if (i == 0):
# 				start_v_index = 0
# 				end_v_index = slice_height+(vert_buffer*2)
# 			elif (i == (vert_slices-1)):
# 				start_v_index = (i*slice_height)-(vert_buffer*2)
# 				#end_v_index = origH
# 				end_v_index = start_v_index + slice_height + (vert_buffer*2)
# 			else:
# 				start_v_index = (i*slice_height)-vert_buffer
# 				end_v_index = ((i+1)*slice_height)+vert_buffer

# 			if (j == 0):
# 				start_h_index = 0
# 				end_h_index = slice_width+(horiz_buffer*2)
# 			elif (j == (horiz_slices-1)):
# 				start_h_index = (j*slice_width)-(horiz_buffer*2)
# 				#end_h_index = origW
# 				end_h_index = start_h_index + slice_width + (horiz_buffer*2)
# 			else:
# 				start_h_index = (j*slice_width)-horiz_buffer
# 				end_h_index = ((j+1)*slice_width)+horiz_buffer

# 			# start_v_index = (0 if (i==0) else (i*slice_height)-vert_buffer)
# 			# end_v_index = (origH if (i == (vert_slices-1)) else (((i+1)*slice_height)+vert_buffer))

# 			# start_h_index = (0 if (j==0) else (j*slice_width)-horiz_buffer)
# 			# end_h_index = (origW if (j == (horiz_slices-1)) else (((j+1)*slice_width)+horiz_buffer))

# 			sliced_image = image[start_v_index:end_v_index, start_h_index:end_h_index, :]

# 			output_images.append(sliced_image)

# 	return output_images

def process_image(image_read, image_real, east, min_confidence, width, height, hyst_X=0, hyst_Y=0, offset_X=0, offset_Y=0, remove_boxes=False):

	#unnecessary default
	args = {"image":"/Users/surajmenon/Desktop/findocDocs/apple_test1.png", "east":"/Users/surajmenon/Desktop/findocDocs/frozen_east_text_detection.pb", "min_confidence":0.5, "width":320, "height":320}

	args['image'] = image_real
	args['east'] = east
	args['min_confidence'] = min_confidence
	args['width'] = width
	args['height'] = height

	if (image_read == True):
		image = cv2.imread(args['image'])
	else:
		image = args['image']

	#print ('Processing Image')
	#print (image.shape)
	print ('.')

	#Saving a original image and shape
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	# print ('Image Size')
	# print (origH)
	# print (origW)
	# exit()

	# set the new height and width to default 320 by using args #dictionary.  
	(newW, newH) = (args["width"], args["height"])

	#Calculate the ratio between original and new image for both height and weight. 
	#This ratio will be used to translate bounding box location on the original image. 
	rW = origW / float(newW)
	rH = origH / float(newH)

	# resize the original image to new dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# construct a blob from the image to forward pass it to EAST model
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)


	net = cv2.dnn.readNet(args["east"])

	# We would like to get two outputs from the EAST model. 
	#1. Probabilty scores for the region whether that contains text or not. 
	#2. Geometry of the text -- Coordinates of the bounding box detecting a text
	# The following two layer need to pulled from EAST model for achieving this. 
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(boxes, confidence_val) = predictions(scores, geometry, args['min_confidence'])
	boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
	
	#connect nearby boxes in the x-direction
	#boxes = connect_horizontal_boxes(boxes, x_threshold=10, y_threshold=10)

	#extra box removal
	# if (remove_boxes==True):
	# 	boxes = remove_nearby_boxes(boxes, threshold=10)

	##Text Detection and Recognition 

	# initialize the list of results
	results = []

	count = 0

	extra_distance = 1

	x_start_buffer = 20 #TODO: FIX, hack since the vision tends to underestimate the start, so shift it 5

	adjusted_boxes = []
	# loop over the bounding boxes to find the coordinate of bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
		startX = int(startX * rW) - hyst_X - x_start_buffer
		startY = int(startY * rH) - hyst_Y 
		endX = int(endX * rW) + hyst_X 
		endY = int(endY * rH) + hyst_Y 

		#bound the bound
		if (startX < 0):
			startX = 0
	   
		if (startY < 0):
			startY = 0

		if (endX > origW):
			endX = origW-1
		if (endY > origH):
			endY = origH-1

		adjusted_box = (startX, startY, endX, endY)
		adjusted_boxes.append(adjusted_box)

	adjusted_boxes = connect_horizontal_boxes(adjusted_boxes, x_threshold=10, y_threshold=10) 

	for (startX, startY, endX, endY) in adjusted_boxes:
		#extract the region of interest
		r = orig[startY:endY, startX:endX]

		#configuration setting to convert image to string.  
		#configuration = ("-l eng --oem 1 --psm 8")
		configuration = ("-l eng --oem 1 --psm 7")
	    ##This will recognize the text from the image of bounding box


		try:
			text = pytesseract.image_to_string(r, config=configuration)
		except:
			print ('Some bounding box out of order')
			text = 'GHAJEFKJEKAFJEKFAJEFKEJKFAEK'


		#shift the coordinates before returning
		# startX += (offset_X*extra_distance)
		# startY += (offset_Y*extra_distance)
		# endX += (offset_X*extra_distance)
		# endY += (offset_Y*extra_distance)

		#print (((startX, startY, endX, endY), text))

		# append bbox coordinate and associated text to the list of results 
		results.append(((startX, startY, endX, endY), text))
		count += 1

	return orig, results
	#return orig, results_m

def show_image(image, results):

	#Display the image with bounding box and recognized text
	#orig_image = orig.copy()
	orig_image = image.copy()

	# Moving over the results and display on the image
	for ((start_X, start_Y, end_X, end_Y), text) in results:
		# display the text detected by Tesseract
		print("{}\n".format(text))

		# Displaying text
		text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
		cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
			(0, 0, 255), 2)
		cv2.putText(orig_image, text, (start_X, start_Y - 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

	plt.imshow(orig_image)
	plt.title('Output')
	plt.show()