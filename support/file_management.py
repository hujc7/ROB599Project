import os
from glob import glob
files = glob('trainval/*/*_image.jpg')

for i in range(len(files)):
	source = files[i]
	destination = "deploy/traindata/" + str(i) + ".jpg"
	os.rename(source,destination)
	source = source.replace("_image.jpg","_label.txt")
	destination = destination.replace(".jpg",".txt")
	os.rename(source,destination)

