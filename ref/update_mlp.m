function [model] = update_mlp(model, input, target)
% this function is called once for every pattern presentation, weights are
% updated every time, which is the magical step.

% this holds the activation of every neuron in every layer
% length(model.weights) == nTransitions

    activations = cell(length(model.weights)+1,1);
    activations{1} = input;
    
    % this loop calculates the activations of all the neuron layers
    for i = 1:length(model.weights)

        % activations{i} is a row vector
        % model.weights{i} is a matrix of weights
        % the output of that product is a row vector of length equal to the
        % number of neurons in the next layer
)
	
	temp = activations{i} * model.weights{i} + model.biases{i}
        activations{i+1} = 1./(1+exp(-(temp))); % squash the output a bit
    end

    % variable for holding the errors at each level
    errors = cell(length(model.weights),1);
    
    % this code propagates the error back through the neural net
    run_error = (target - activations{end}); %keeps track of the error at each loop
    for i = length(model.weights):-1:1
        %Z=activations{i+1}
        %one_minus_Z=(1-activations{i+1})
        %Zd_minus_Z = (run_error)
        
        errors{i} = activations{i+1} .* (1-activations{i+1}) .* (run_error);
        %delta = errors{i}
        run_error = errors{i} * model.weights{i}';
    end
    
    % this code updates the weights and biases
    for i = 1:length(model.weights)
        % update weights based on the learning rate, the input activation
        % and the error
        model.weights{i} = model.weights{i} + model.learning_rate * activations{i}' * errors{i};
        % update the neuron biases as well
        model.biases{i} = model.biases{i} + model.learning_rate * errors{i};
        % it takes a while to figure out all the matrix operations, but
        % once it's done it's nice.
    end
end
