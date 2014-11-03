function [output cc] = test_mlp(model, input, target)
    % this function calculates the output of the model, and compares it to the
    % target output
    [ntest nOutLayer] = size(target);
    
    % this variable is for the final output of the neural net
    output = zeros(ntest, nOutLayer);
    for i = 1:ntest
        temp = input(i,:); % output at each layer, gets updated
        for j = 1:length(model.weights)
            temp = temp * model.weights{j} + model.biases{j}; % calculate the output
            temp = 1./(1+exp(-temp)); % squashit
        end
        output(i,:) = temp; % keep only the last output value
    end
    warning('off', 'all') % corrcoef gives some divide by zero errors, this is the laziest fix possible
    cc = corrcoef(target(:), output(:));
    if(numel(cc)>1) % Octave and MATLAB do corrcoef slightly differently, so this is to make things consistent
        cc = cc(2,1); 
    end
end