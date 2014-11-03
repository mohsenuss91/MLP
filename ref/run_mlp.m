tic

%% set of data for learning some populations in 2D space

% this code generates some synthetic data, it consists of 2 dimensional
% data from 6 populations in this case, they are partially overlapping with
% the parameters used here.
% There are plots at the end that make it clear what the populations are
% and how the learned model performs on an independent test set.  The third
% parameter set to 1, makes the problem not linearly separable, requiring a
% hidden layer in the network to succeed.
[training training_class testing testing_class] = clouds_on_unit_circle(6, .25, 1);

% 5 neurons in the hidden layer performs well most of the time, sometimes
% it fails on one class, but usually it works
hidden_layers = [5];
iterations = 500;  
learning_rate = 0.1;
momentum = 0.1;

% initialize and train the model
[model cc_train output_train] = train_mlp(training, training_class, hidden_layers, iterations, learning_rate, momentum);



%% test the model on the independent test set
%% output is the output of the model, and the cross correlation of the
%% output to the target output
%[output_test cc_test] = test_mlp(model, testing, testing_class);
%
%% estimate class from the output as the unit of maximal activation
%[jnk train_class] = max(training_class'); % for the traianing data
%[jnk true_class] = max(testing_class');   % for the testing data
%[jnk est_class] = max(output_test');      % the decision of the model
%
%%% calculate percent correct
%[ntest nclasses] = size(testing_class);
%classification_errors = true_class ~= est_class;
%percent_correct = 100 * (1 - sum(classification_errors) / ntest)
%
%%% calculate the confusion matrix
%confusion = zeros(nclasses);
%for i = 1:ntest
%    confusion(true_class(i), est_class(i)) = confusion(true_class(i), est_class(i)) + 1;
%end
%confusion
%
%%% make plots of the training and testing data
%figure, 
%colors = 'rgbcmyrgbcmyrgbcmyrgbcmyrgbcmyrgbcmyrgbcmy';
%for i = 1:nclasses
%    subplot(1,2,1); hold on; axis equal;
%    % plot the training data, color coded by the class
%    plot(training(train_class==i,1), training(train_class==i,2), [colors(i) 'o']);
%    title('training data');
%    subplot(1,2,2); hold on; axis equal;
%    % plot the testing data, color coded by the estimated class
%    plot(testing(est_class==i,1), testing(est_class==i,2), [colors(i) 'o']);
%    title('testing data');
%end
%% indicate on the testing data where the errors are
%plot(testing(classification_errors,1), testing(classification_errors,2), 'kx');
%
%%% calculate and plot the decision boundary
%
%% make a grid of inputs
%xpts = 0:.05:4;
%ypts = 0:.05:4;
%gridinput = zeros(length(xpts)*length(ypts),2);
%for i =1:length(xpts) % all these loops are just for generating the input, scattered on a 2D grid
%    for j = 1:length(ypts) % probably MATLAB has an easier way to do it
%        gridinput(i+(j-1)*length(ypts),1) = xpts(i);
%        gridinput(i+(j-1)*length(ypts),2) = ypts(j);
%    end
%end
%
%% this is where the grid is shown to the model, to get its estimate of class
%[output] = test_mlp(model, gridinput, zeros(length(xpts)*length(ypts),nclasses));
%[jnk grid_est] = max(output');   
%
%% display the estimates on the 2D plane to see the decision boundaries
%figure, hold on;
%for i = 1:nclasses
%    plot(gridinput(grid_est==i,1), gridinput(grid_est==i,2),[colors(i) '*']);
%end
%plot(testing(:,1),testing(:,2), 'k.');
%title('Decision boundaries for the learned model');
%
%%% learning text classifications
%% the only difference here is that the data starts as text classes and has
%% to be preprocessed first
%veg = {{'pumpkin'};{'pumpkin'};{'carrot'};{'zucchini'};{'zucchini'};{'yam'};{'potato'}};
%data = {{'brown','yellow','large'};
%	{'brown','yellow','small'}; 
%	{'brown','yellow','underground','small'};
%	{'green','small'};
%	{'green','large'};
%	{'brown','underground'};
%	{'brown','underground','small'}};
%
%% preprocess data, make text into binary attributes
%input_strings = data{1};
%output_strings = veg{1};
%for i = 2:length(data)
%    input_strings = [input_strings data{i}];
%    output_strings = [output_strings veg{i}];
%end
%
%% find out how many different input and output values there are
%input_strings = unique(input_strings);
%output_strings = unique(output_strings);
%
%% for each input and output value, make a binary value for each of them
%binary_data = zeros(length(data), length(input_strings)); % holds the binary input 
%binary_veg = zeros(length(data), length(output_strings)); % holds the binary output
%
%for i = 1:length(data)
%    for j = 1:length(data{i}) % find the corresponding binary value and set it to 1
%        binary_data(i,:) = binary_data(i,:) + strcmp(input_strings, data{i}{j});
%    end
%    for j = 1:length(veg{i}) % as above
%        binary_veg(i,:) = binary_veg(i,:) + strcmp(output_strings, veg{i}{j});
%    end
%end
%% all that was to make the string array into a binary array
%
%% this is where the actual mlp will be generated and trained
%hidden_layers = []; % learning in this dataset works with no hidden layers
%iterations = 500;   % number of iterations through the entire training set
%learning_rate = 0.5;% learning rate of the algorithm
%momentum = 0.1;
%
%% give the binary input to the algorithm and watch the magic
%[model cc_train output_train] = train_mlp(binary_data, binary_veg, hidden_layers, iterations, learning_rate, momentum);
%% estimate the class from the output of the model, as the index of the
%
%% neuron with the max activation
%[maxes indxs] = max(output_train, [], 2);
%class_estimates = output_strings(indxs);
%
%% count up number of correct answers, and display the output
%nCorrect = 0;
%for i = 1:length(veg)
%    target = veg{i}{1}; 
%    output = class_estimates{i};
%    fprintf('target = {%s} ::: model output = {%s}\n ', target, output);
%    if(strcmp(target,output)==1)
%        nCorrect = nCorrect + 1;
%    end
%end
%% percent correct
%percent_correct_on_training_set = 100* (nCorrect / length(veg))
%
%% cross correlation of target with output
%cc_train
%
%toc
