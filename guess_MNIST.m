function [ output] = guess_MNIST(index,test_input,test_labels,Act,W,B,num_layer)

 [guess_result] = FP(test_input{index},Act,W,B,num_layer);
  
 
 
[grow,gcol]=find(guess_result{end}==max(guess_result{end}));
        
        
 output = [test_labels(index),gcol-1];
 
 imshow(reshape(test_input{index},sqrt(length(test_input{index})),sqrt(length(test_input{index}))))
end

