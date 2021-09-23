import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
import copy
from detect_prod import get_boxes_from_model

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

def translate_boxes(pre_boxes):
	#assume our input is in x1, y1, x2, y2, ..
	#we want to output in min_x, min_y, max_x, max_y

	num_boxes = pre_boxes.shape[0]
	new_boxes = np.zeros((num_boxes, 4))
	for i in range(num_boxes):
		c_box = pre_boxes[i, :]
		new_boxes[i, 0] = np.amin(c_box[[0, 2, 4, 6]])
		new_boxes[i, 1] = np.amin(c_box[[1, 3, 5, 7]])
		new_boxes[i, 2] = np.amax(c_box[[0, 2, 4, 6]])
		new_boxes[i, 3] = np.amax(c_box[[1, 3, 5, 7]])


	return new_boxes

def process_image(img_path, model_path, min_confidence, hyst_X=0, hyst_Y=0, offset_X=0, offset_Y=0, remove_boxes=False, scale=2, model='EASTER'):

	boxes = get_boxes_from_model(img_path, model_path, scale=scale, model=model)
	boxes = translate_boxes(boxes)

	print (boxes.shape)
	exit()
	
	##Text Detection and Recognition 

	# initialize the list of results
	results = []
	
	#for now, say we don't want any X-shifting
	x_start_buffer = 0

	boxes = connect_horizontal_boxes(boxes, x_threshold=45, y_threshold=20) 
	adjusted_boxes = []

	# loop over the bounding boxes to find the coordinate of bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
		startX = int(startX * rW) - hyst_X - x_start_buffer
		startY = int(startY * rH) - hyst_Y 
		endX = int(endX * rW) + hyst_X  - x_start_buffer
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

	#adjusted_boxes = connect_horizontal_boxes(adjusted_boxes, x_threshold=5, y_threshold=15) 

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

		# append bbox coordinate and associated text to the list of results 
		results.append(((startX, startY, endX, endY), text))

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


if __name__ == '__main__':
	#img_path = '/Users/surajmenon/Desktop/findocDocs/apple_tc_full1.png'
	img_path = 'EAST-master-torch/test_img/' + 'apple_tc_full1' + '.jpg'
	model_path  = './EAST-master-torch/pths2/EASTER-sm1-aug3-no_ignore-450.pth'
	min_confidence = 0.9

	process_image(img_path, model_path, min_confidence, hyst_X=0, hyst_Y=0, offset_X=0, offset_Y=0, remove_boxes=False, scale=2, model='EASTER')
