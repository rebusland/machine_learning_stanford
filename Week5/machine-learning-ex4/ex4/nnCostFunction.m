function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% represent the y vector, observed outcomes a m x K matrix, each row is a vector [0 .. 1 .. 0], where
% the index of element 1 represent the class (index 0 is for the label 10)
y = ([1:num_labels] == y);

% add bias column to X
X = [ones(m,1) X];

% input + 1 hidden_layer_size + output
num_layers = 3;

% vector of parameters for the network to matrices per layer
network_matrices = {Theta1, Theta2};

% initialize accumulators for delta term (for gradient calculation)
deltas_accum = {};
for idx = 1 : length(network_matrices)
	deltas_accum{idx} = zeros(size(network_matrices{idx}));
end


% value of cost function for each training sample
costs = zeros(m,1);

% activation nodes matrices: the i-th element of the cell array should be a matrix
% of the i-th activation layer for each training sample.
A = {};

% first activation nodes are the inputs
% transpose it so that each column is an input sample (we have thus 1..m columns) 
A{1} = X'; % n x m matrix (m training sample, n number of input features: e.g. pixel grey-scale)

% 
% Forward propagation
%
% move through the layers
for j = 1 : num_layers - 1
	z = cell2mat(network_matrices(j)) * A{j};
	A{j+1} = 1 ./ (1 + exp(-z));
	if j + 1 != num_layers
		A{j+1} = [ones(1,columns(A{j+1})); A{j+1}]; % add bias nodes
	end
end

% compute cost function term for each training sample
% the sum(..,2) is for making row * column multiplication on the two matrices and get a column vector
costs = sum(-y .* log(A{num_layers})' -(1 - y) .* log(1 - A{num_layers})', 2);

%
% Back-propagation
%
% deltas in each layer
deltas = {};

L = num_layers;

% difference (outcome - predicted) at the output node
deltas{L} = A{L} - y'; % NB y row is transposed to get vertical vectors like [0; ..1; ..;0] for each training sample

% add ficticious "bias" to delta to make the below code to work even for the output layer
deltas{L} = [ones(1,columns(deltas{L})); deltas{L}];

% go backward (up to the first hidden node)
while L > 1
	deltas{L - 1} = (network_matrices{L - 1})' * deltas{L}(2:end, :) .* A{L - 1} .* (1 - A{L - 1});
	deltas_accum{L - 1} += deltas{L}(2:end, :) * (A{L - 1})'; % NB the delta_0 component is discarded as it refers to the bias node
	L -= 1;
end

regularization_term = 0;
for kk = 1 : length(network_matrices)
	matr = network_matrices{kk};
	matr(:,1) = 0; % ignore "bias" terms
	regularization_term += sum(sum(matr .^ 2));
end

overall_cost = sum(costs);
J = (overall_cost / m) + (0.5 * lambda / m) * regularization_term;


% -------------------------------------------------------------

Jacobian = [];
for s = 1 : num_layers - 1
	jacobian_regularization = lambda * network_matrices{s};
	jacobian_regularization(:,1) = 0; % "bias" column must not influence regularization
	jacobian_matrix = (1 / m) * (deltas_accum{s} + jacobian_regularization);

	% unroll to a single column array
	Jacobian = [Jacobian; jacobian_matrix(:)];
end

% =========================================================================

% Unroll gradients
% grad = [Theta1_grad(:) ; Theta2_grad(:)];

grad = Jacobian;


end
