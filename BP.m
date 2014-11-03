function [W,B,Err] = BP(train_output,Act,W,B,num_layer,lrn_rate,Err)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here


prev_err = (train_output - Act{end});
for k=num_layer-1:-1:1
    Err{k} = Act{k+1}.*(1-Act{k+1}).*prev_err;
    prev_err = Err{k} * W{k}';
end

for k=1:num_layer-1
    W{k} = W{k}+lrn_rate*Act{k}'*Err{k};
    B{k} = B{k}+lrn_rate*Err{k};
    
end

end

