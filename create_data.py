# Creating database
# It captures images and stores them in datasets
# folder under the folder name of sub_data
import cv2, sys, numpy, os
import time
haar_file = 'haarcascade_frontalface_default.xml'

# All the faces data will be
# present this folder
datasets = 'datasets'


# These are sub data sets of folder,

sub_data = input("Name: ")	

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
	os.mkdir(path)

# defining the size of images
(width, height) = (130, 100)	

#'0' is used for my webcam,
# if you've any other camera
# attached use '1' like this
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

# The program loops until it has 120 images of the face.
count = 1
angles = 0
ans = "."
print("Face Webcam head on")
print("Be still")
print("Starting in 3 seconds...")
time.sleep(3)
while count < 120:
	if count == 0:
		angles += 1
		ans = input("Face head on.")
	if count == 24:
		angles += 1
		ans = input("Face Left.")
	if count == 48:
		angles += 1
		ans = input("Face Right.")
	if count == 72:
		angles += 1
		ans = input("Face Down.")
	if count == 96:
		angles += 1
		ans = input("Face up.")
	if angles == 1:
		print("Face head on.")
	if angles == 2:
		print("Face Left.")
	if angles == 3:
		print("Face Right.")
	if angles == 4:
		print("Face Down.")
	if angles == 5:
		print("Face up.")
	(_, im) = webcam.read()
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 4)
	for (x, y, w, h) in faces:
		cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
		face = gray[y:y + h, x:x + w]
		face_resize = cv2.resize(face, (width, height))
		cv2.imwrite('% s/% s.png' % (path, count), face_resize)
	print(count)
	count += 1
	
	cv2.imshow('OpenCV', im)
	key = cv2.waitKey(10)
	if key == 27:
		break