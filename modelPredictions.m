function [classesPredictions,classCorr] = modelPredictions(dlnet,mbq,classes)

    classesPredictions = [];    
    classCorr = [];  
    
    while hasdata(mbq)
        [dlX1,dlX2,dlY] = next(mbq);
        
        % Make predictions using the model function.
        dlYPred = predict(dlnet,dlX1,dlX2);
        
        % Determine predicted classes.
        YPredBatch = onehotdecode(dlYPred,classes,1);
        classesPredictions = [classesPredictions YPredBatch];
                
        % Compare predicted and true classes.
        Y = onehotdecode(dlY,classes,1);
        classCorr = [classCorr YPredBatch == Y];
                
    end

end