function [training training_class testing testing_class] = clouds_on_unit_circle(nclasses, noise, make_hard)
% this is just a helper function to make some synthetic data for me
% it makes 2D data with a given number of populations.  If you select the
% option to 'make_hard' the problem, it makes the populations, and then
% combines some of them, making the solution not linearly
% separable, requiring a hidden layer to effectively
% classify the data
ntraining_per_class = 100;
ntraining = ntraining_per_class*nclasses;

ntest_per_class = 100;
ntest = ntest_per_class*nclasses;

% the centroids of the populations to be learned will be
% equally spaced around the unit circle
angles = linspace(0,2*pi,nclasses + 1);
xs = cos(angles);
ys = sin(angles);
xoff = 2; % offset from zero (requires proper coding of the bias output to work in that case)
yoff = 2;

% create variable for the training set
training = zeros(ntraining, 2); % input is 2 dimensional
training_class = zeros(ntraining, nclasses); % output is nclasses dimensional binary 

% create variable for the testing set
testing = zeros(ntest, 2);
testing_class = zeros(ntest, nclasses);

% initialize the code for the training and testing sets
for i = 1:nclasses
    for j = 1:ntraining_per_class
        row = (i-1)*ntraining_per_class+j;
        training(row,1) = xoff + xs(i)+noise*randn();
        training(row,2) = yoff + ys(i)+noise*randn();
        training_class(row,i) = 1;
    end
    for j = 1:ntest_per_class
        row = (i-1)*ntest_per_class+j;
        testing(row,1) = xoff + xs(i)+noise*randn();
        testing(row,2) = yoff + ys(i)+noise*randn();
        testing_class(row,i) = 1;
    end
end

if(make_hard)
    % make the problem harder by not being linearly separable
    training_class(:,2) = training_class(:,2) + training_class(:,4);
    training_class(:,3) = training_class(:,3) + training_class(:,5);
    training_class(:,4:5) = [];
    testing_class(:,2) = testing_class(:,2) + testing_class(:,4);
    testing_class(:,3) = testing_class(:,3) + testing_class(:,5);
    testing_class(:,4:5) = [];

    nclasses = nclasses - 2;
end

end
