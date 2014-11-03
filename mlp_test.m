


%% MLP 2-layer to test XOR
clear;
clc;

Mode = 'MNIST'
%Mode = 'XOR'

if (strcmp(Mode,'MNIST'))
    % Load the digits into workspace (MNIST Test, from
    % http://yann.lecun.com/exdb/mnist/)
    num_train = 100;
    [train_IMG,train_labels,test_IMG,test_labels] = readMNIST(num_train);
    input =cell(num_train,1);
    output =cell(num_train,1);
    
    test_input=cell(length(test_IMG),1);
    test_output=cell(length(test_labels),1);
    
    for i=1:num_train
        input_img = train_IMG{i};
        [width height] = size(input_img);
        img_vec = reshape(input_img,1,width*height);
        input{i}=double(img_vec);
        output{i} = double(train_labels(i));
    end
    
    for i=1:length(test_input)
        input_img = test_IMG{i};
        [width height] = size(input_img);
        img_vec = reshape(input_img,1,width*height);
        test_input{i} = double(img_vec);
        test_output{i} = double(test_labels(i));
    end
elseif (strcmp(Mode,'XOR'))
    
    num_train = 4;
    input =cell(num_train,1);
    output =cell(num_train,1);
    input{1} = [0 0];
    input{2} = [0 1];
    input{3} = [1 0];
    input{4} = [1 1];
    output{1} = [0];
    output{2} = [1];
    output{3} = [1];
    output{4} = [0];
    
end

if (length(input)~=length(output))
    error('len_input does not equal to len_output');
end





%% Global Var??
% global train_IMG;
% global train_labels;
% global test_IMG;
% global test_labels;
% global num_layer;
% global W;
% global B;
% global Act;




%% Determine # of nodes in hidden layer & output layer
num_node_il = length(input{1});
num_node_hl = [num_node_il*2];
num_node_ol = length(output{1});

set_node =[num_node_il num_node_hl num_node_ol];

%% Init. template (random)
rand('state',sum(100*clock));

num_layer = length(set_node);
W=cell(num_layer-1,1);
B=cell(num_layer-1);

for i=1:num_layer-1
    W{i} = -1+2.*rand(set_node(i),set_node(i+1));
    B{i} = rand(1,set_node(i+1));
end

%% Learning coeff = 0.7 & Iteration = 10
lrn_rate = 0.5;
max_iter = 10000;


tic


Act=cell(num_layer,1);
Err=cell(num_layer-1,1);
for i= 1:max_iter
    for j= 1:num_train
        train_input = input{j};
        train_output = output{j};
        
        
        % Forward Propagation
        [Act]   =   FP(train_input,Act,W,B,num_layer);
        % Backward Propagation & Template update
        [W,B,Err]       =   BP(train_output,Act,W,B,num_layer,lrn_rate,Err);
        
        Result(i).W=W;
        Result(i).B=B;
        Result(i).Err_1 = Err{1};
        Result(i).Err_2 = Err{2};
    end
end

toc
disp('Training Ends')

if (strcmp(Mode,'XOR'))
    grid = [0:0.01:1];
    Z=-1*ones(length(grid),length(grid));
    
    for i=1:length(grid)
        for j=1:length(grid)
            test = [grid(i) grid(j)];
            Act_new = FP(test,Act,W,B,num_layer);
            Z(i,j) = Act_new{3};
            
        end
    end
end
[X,Y] = meshgrid(grid);
mesh(X,Y,Z)


