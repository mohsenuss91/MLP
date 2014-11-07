function [Act] = FP(train_input, Act,W,B,num_layer)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
Act{1} = train_input;
for k=1:num_layer-1

    
    temp = Act{k} * W{k} + B{k};
    Act{k+1} = active_func(temp);
    
    
end

end

