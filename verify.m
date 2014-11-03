function [] = verify(index,test_IMG,test_labels,test_input,W,B,num_layer)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

figure(1);imshow(test_IMG{index})
label = test_labels(index)
Act_new=cell(num_layer,1);
Act_new = FP(test_input{index},Act_new,W,B,num_layer);
guess = Act_new{end}
end

