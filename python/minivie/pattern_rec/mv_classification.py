#University of Portland, Shiley School of Engineering
#OHSU Helping Hands Spring 2022
#Authors: Justin Cao, Alex Junkins
#Machine Vision Classification
#This file takes a snapshot with the camera /dev/video0 and classifies it 
#based on the given model at modelDir using the jetson inference library
#Version: April 1, 2022

import os
import jetson.inference
import jetson.utils

import argparse

class MachineVision:
	@staticmethod
	def classifyImage():
		#commandline variables
		modelDir = "classifier_data/resnet18.onnx"
		    #original location: models/eatware/resnet18.onnx
		input_blob = "input_0"
		output_blob = "output_0"
		labels = "classifier_data/labels.txt"
		    #original location: data/eatware_data/labels.txt
		camera = "/dev/video0"
		cam_width = "640"
		cam_height = "480"
		cam_resolution = cam_width+"x"+cam_height

		origImgName = "classify_me.jpg"

		#snapshot the camera feed and store the image
		snapshot_command = "fswebcam -d "+camera+" -r "+cam_resolution+" --no-banner " + origImgName
		os.system(snapshot_command)

		#load stored image (into shared CPU/GPU memory)
		img = jetson.utils.loadImage(origImgName)

		#load the recognition model
		model = jetson.inference.imageNet(argv=['--model='+modelDir, '--labels='+labels, '--input_blob='+input_blob, '--output_blob='+output_blob])

		#classify the image
		class_idx, confidence = model.Classify(img)

		#find the object description
		class_desc = model.GetClassDesc(class_idx)

		#print out the result
		print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
		
		#return values for use in MINIVIE
		return class_desc, (confidence*100)



def printTest():
	print("Hello, world!")

#MachineVision.classifyImage()
