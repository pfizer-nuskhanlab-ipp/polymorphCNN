%% Classify image data with trained Convolution Neural Network

% load trained convolutional neural network for classification of images of
% alpha and beta glycine crystals
clear;clc;close all;
load('polymorphCNNtrainedNetwork.mat','dlnet','classes','miniBatchSize');

% Specify the directory in which images you want to classify are stored


dataDir1=fullfile('C:\Users\ariel\Desktop\glycinepolymorphs\beta');% directory
imdstest = imageDatastore(dataDir1);
num_images=numel(imdstest.Files);
polarizerangletest=zeros(num_images,1); % we evaluate test images at polarizer angles of 0 degrees
boguslabels(1:(num_images-2),1)=categorical("alpha"); % bogus labels just to make the code run
boguslabels((num_images-1):num_images,1)=categorical("beta");

dsimagedatatest=imdstest;
dsangledatatest=arrayDatastore(polarizerangletest);
dslabeldatatest=arrayDatastore(boguslabels);
dsTest=combine(dsimagedatatest,dsangledatatest,dslabeldatatest);

tic

mbqTest = minibatchqueue(dsTest,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB','CB',''},'PartialMiniBatch','return');

predictionsTest = modelPredictions(dlnet,mbqTest,classes);
predictionsTest = transpose(predictionsTest);
percentagecount_alpha=mean(predictionsTest=='alpha')*100
percentagecount_beta=mean(predictionsTest=='beta')*100

toc