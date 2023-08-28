%% training network for alpha and beta glycine classification

close all; clear;clc;
tic

% Specify the directory in which training images are stored.
% In this directory we have 2 folders alpha and beta 
dataDir=fullfile('D:\training_polarizerangle_luminance_1.2');
load alpha alphaglycineDataset
load beta betaglycineDataset

% load data for polarizer angles 
alphaglycinepolarizerangle1=alphaglycineDataset.singlecrystalpolarizerangle;
betaglycinepolarizerangle1=betaglycineDataset.singlecrystalpolarizerangle;

% 7 sets of images acquired at various luminance factors therefore we
% replicate the polarizer angles 7 times
alphaglycinepolarizerangle=[alphaglycinepolarizerangle1;alphaglycinepolarizerangle1;alphaglycinepolarizerangle1;alphaglycinepolarizerangle1;alphaglycinepolarizerangle1;alphaglycinepolarizerangle1;alphaglycinepolarizerangle1];
betaglycinepolarizerangle=[betaglycinepolarizerangle1;betaglycinepolarizerangle1;betaglycinepolarizerangle1;betaglycinepolarizerangle1;betaglycinepolarizerangle1;betaglycinepolarizerangle1;betaglycinepolarizerangle1];
combinedpolarizerangle=[alphaglycinepolarizerangle;betaglycinepolarizerangle];
[r1,c1]=size(alphaglycinepolarizerangle);
[r2,c2]=size(betaglycinepolarizerangle);

imds = imageDatastore(dataDir, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% split image datastore into training and validation data
fractiontraindata=0.8;
[imds1,imds2] = splitEachLabel(imds,fractiontraindata);
labeldata1=imds1.Labels;
labeldata2=imds2.Labels;

% training data
numtrainalphabeta1=countEachLabel(imds1);
numtrainalpha1=numtrainalphabeta1.Count(1);
numtrainbeta1=numtrainalphabeta1.Count(2);

angledata1=[alphaglycinepolarizerangle(1:numtrainalpha1);betaglycinepolarizerangle(1:numtrainbeta1)];

dsimagedata1=imds1;
dsangledata1=arrayDatastore(angledata1);
dslabeldata1=arrayDatastore(labeldata1);
dsTrain=combine(dsimagedata1,dsangledata1,dslabeldata1);

% validation data
angledata2=[alphaglycinepolarizerangle((numtrainalpha1+1):r1);betaglycinepolarizerangle((numtrainbeta1+1):r2)];

dsimagedata2=imds2;
dsangledata2=arrayDatastore(angledata2);
dslabeldata2=arrayDatastore(labeldata2);
dsValidation=combine(dsimagedata2,dsangledata2,dslabeldata2);

% display random images of training images
numTrainImages = numel(labeldata1);
figure
idx = randperm(numTrainImages,20);
for i = 1:numel(idx)
    subplot(4,5,i)    
    imshow(imread(imds.Files{i,1}))
    title("Angle: " + angledata1(idx(i)))
end

% Calculate the number of images in each category. labelCount is a table that contains the labels and the number of images having each label.
labelCount = countEachLabel(imds)
classes=categories(imds.Labels);


% Specify the size of the images in the input layer of the network
img = readimage(imds,1);
[height, width, channel]=size(img);

% Define Network Architecture
layers1 = [
    imageInputLayer([height width channel],'Normalization','none','Name','images')
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','BatchNormalization_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2,'Stride',2,'Name','Maxpooling_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BatchNormalization_2')
    reluLayer('Name','relu_2')
    maxPooling2dLayer(2,'Stride',2,'Name','Maxpooling_2')
    
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')
    batchNormalizationLayer('Name','BatchNormalization_3')
    reluLayer('Name','relu_3')
    fullyConnectedLayer(2,'Name','fullyconnectedlayer1')
    concatenationLayer(1,2,'Name','concat')
    fullyConnectedLayer(2,'Name','fullyconnectedlayer2')
    softmaxLayer('Name','softmax')
    ];

lgraph=layerGraph(layers1);

featInput = featureInputLayer(1,'Name','features');
lgraph = addLayers(lgraph, featInput);
lgraph = connectLayers(lgraph, 'features', 'concat/in2');

figure;
plot(lgraph);
title('Network Architecture');

% create dlnetworkobject
dlnet=dlnetwork(lgraph);
dlnet.InputNames

% model training options user can edit this section as they please
numEpochs = 1;
miniBatchSize = 128;

learnRate = 0.0001;
momentum = 0.9;

%Training model
velocity = [];

mbq = minibatchqueue(dsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB','CB',''});

mbqValidation = minibatchqueue(dsValidation,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat',{'SSCB','CB',''},'PartialMiniBatch','return');

% set plot colors
    figure
    lineLossTrain = animatedline('Color','red');
    
    lineLossValidation = animatedline( ...
    'LineStyle','--', ...
    'Color','black');
    
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on

    figure
    lineAccuracyTrain = animatedline('Color','blue');
    
    lineAccuracyValidation = animatedline( ...
    'LineStyle','--', ...
    'Color','black');
    
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Accuracy")
    grid on
    
iteration = 0;
counter=1;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbq)
    shuffle(mbqValidation)
    % Loop over mini-batches.
    while hasdata(mbq)

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [dlX1,dlX2,dlY] = next(mbq);
        
        if (iteration*miniBatchSize) > (counter*length(angledata2)) && ((iteration-1)*miniBatchSize)< (counter*length(angledata2))
           shuffle(mbqValidation)
           counter=counter+1;
        end
        
        [dlX1V,dlX2V,dlYV]=next(mbqValidation);
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss,accuracy] = dlfeval(@modelGradients,dlnet,dlX1,dlX2,dlY);
        [gradientsV,stateV,lossV,accuracyV]=dlfeval(@modelGradients,dlnet,dlX1V,dlX2V,dlYV);
        
        dlnet.State = state;
        
        % Update the network parameters using the SGDM optimizer.
        [dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        
        % Display the training progress.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        %completionPercentage = round(iteration/numIterations*100,0);
        
        title("Epoch: " + epoch + ", Elapsed: " + string(D));
        addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
        drawnow   
        
        addpoints(lineLossValidation,iteration,double(gather(extractdata(lossV))))
        drawnow 
        
        % Display the training progress.
        D1 = duration(0,0,toc(start),'Format','hh:mm:ss');
        %completionPercentage = round(iteration/numIterations*100,0);
        
        title("Epoch: " + epoch + ", Elapsed: " + string(D1));
        addpoints(lineAccuracyTrain,iteration,accuracy)
        drawnow   
        
        addpoints(lineAccuracyValidation,iteration,accuracyV)
        drawnow 
        lossvalue=double(gather(extractdata(loss)));
        lossvalueV=double(gather(extractdata(lossV)));
        
        datastoreaccuracy(iteration,1)=accuracy;
        datastoreaccuracyV(iteration,1)=accuracyV;
        datastoreloss(iteration,1)=lossvalue;
        datastorelossV(iteration,1)=lossvalueV;

        % save trained CNN at regular checkpoints
        if mod(epoch,2500)==0
            save(sprintf('workspace_epochs_%d',epoch));
        end 
    end
end
save('polymorphCNNtrainedNetwork.mat');


