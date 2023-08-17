function [X,angle,Y] = preprocessMiniBatch(XCell,angleCell,YCell)
    
    % Extract image data from cell and concatenate.
    X = cat(4,XCell{:});
    % Extract angle data from cell and concatenate.
    angle = cat(2,angleCell{:});
    % Extract label data from cell and concatenate.
    Y = cat(2,YCell{:});    
        
    % One-hot encode labels.
    Y = onehotencode(Y,1);
    
end