function [output] = active_func(input)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


%output = 1/(1+e^(input*-1));

output = 1./(1+exp(-1*input));
end

