function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
X = [ones(m, 1) X];

num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

# layer 2, hidden layer
z_layer_2 = X*transpose(Theta1); % 5000 X 25 
a_layer_2 = sigmoid(z_layer_2); % 5000 X 25
a_layer_2 = [ones(m, 1) a_layer_2]; % add bias unit, 5000 X 26 

# layer 3, output layer
z_layer_3 = a_layer_2*transpose(Theta2); % 5000 X 10
a_layer_3 = sigmoid(z_layer_3); % 5000 X 10

% predict for all 10 different models (a set of 10 different theta values) 
% and pick the max probability and the index for which the prob is max
Matrix_Dim = 2; %operate olong 2nd dimension (max from every column)
[prob, prediction] = max(a_layer_3,[],Matrix_Dim);
p = prediction;

% =========================================================================


end
