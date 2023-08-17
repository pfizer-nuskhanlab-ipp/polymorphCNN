function [gradients,state,loss,accuracy] = modelGradients(dlnet,dlX1,dlX2,Y)

[dlYPred,state] = forward(dlnet,dlX1,dlX2);

loss = crossentropy(dlYPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

classes={'alpha';'beta'}; %  change this according to the name of categories you have
YPred=onehotdecode(dlYPred,classes,1);
Y1=onehotdecode(Y,classes,1);
accuracy=mean(YPred==Y1);
end