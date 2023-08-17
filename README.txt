Polymorph convolutional neural network (CNN) 

This convolutional neural network leverages birefringence of crystals for molecular crystal classification.
This code should be applied onto polarized light microscopy (PLM) images of birefringent crystal ensembles acquired under cross-polarized mode.
This is a case of supervised learning.
Detailed explanation of the experimental setup can be found in the reference publication listed below.


This repository contains several matlab scripts:
1. crystalcropper.m
	Crops images of single crystals from tif image labelled '0001.tif', records the polarizer angle at the time of image acquisition and saves the workspace data into a .mat (e.g., alpha.mat or beta.mat) file.
	Cropped images are saved in labelled folders (Refer to INSTRUCTIONS FOR FOLDER LABELLING).
	.mat files should be saved in the folder comprising labelled folders (e.g., C:\Users\USERNAME\Desktop\glycinepolymorphs).
	
	INSTRUCTIONS FOR FOLDER LABELLING:
	(a) For images that will be used for training the CNN: The folder should be labelled with name of the polymorphic form (e.g. alpha or beta in our case).
		mkdir('C:\Users\USERNAME\Desktop\glycinepolymorphs\alpha'), creates a folder with label alpha in the folder glycinepolymorphs located in the desktop
		mkdir('C:\Users\USERNAME\Desktop\glycinepolymorphs\beta'), creates a folder with label beta in the folder glycinepolymorphs located in the desktop
		All cropped images of single crystals of alpha and beta glycine should be written into their respective folders labelled alpha and beta.
	
	(b) For images of birefringent crystal ensembles which you wish to analyze after the CNN has been trained: You can name the folder however you like. 
		mkdir('C:\Users\USERNAME\Desktop\test')
		All cropped images of single crystals will be written into the folder labelled as test located in the desktop
	
	INSTRUCTIONS FOR CODE USE:
	Paste the crystalcropper.m file in the same folder as the images of crystal ensembles you wish to crop. 
	We recommend that the images of crystal ensembles are labelled in ascending order (e.g., 0001.tif, 0002.tif, 0003.tif).
	Since images are acquired at angular intervals of 1.2 degrees, this ensures that all single crystals cropped from image 0001.tif is tagged at polarizer angle 0 degrees (polarization axis) 
	and those from image 0151.tif is tagged at polarizer angle 180 degrees.


2. polymorphCNNtrainer.m
	Loads images of cropped single crystals from specified folder directory (e.g., 'C:\Users\USERNAME\Desktop\glycinepolymorphs').
	Trains a convolutional neural network with 2 input features, namely, single crystal images and polarizer angles at which the images are acquired.
	Saves the workspace data into a .mat file after CNN training is completed (e.g., polymorphCNNtrainedNetwork.mat).

3. polymorphCNNclassifier.m
	Loads the trained network from the .mat file (e.g.,polymorphCNNtrainedNetwork.mat)
	Classifies the polymorphic form of the single crystals in the test folder (e.g., 'C:\Users\USERNAME\Desktop\glycinepolymorphs\alpha')
	

You should use the scripts in the following sequence:
1. Use crystalcropper to extract images of single crystals and record polarizer angle data   
2. Use polymorphCNNtrainer to train CNN for classification of polymorphs 
3. Use polymorphCNNclassifier after CNN training has been completed to classify the polymorphs of single crystals in the test folder


Reference publication: 
Harnessing Birefringence for Real-time Classification of Molecular Crystals using Dynamic Polarized Light Microscopy, Microfluidics and Machine Learning.
Codes written by Dr Ariel Y. H. Chua in collaboration with Dr Eunice W. Q. Yeap , Dr David M. Walker , Dr Joel M. Hawkins , Dr Saif A. Khan