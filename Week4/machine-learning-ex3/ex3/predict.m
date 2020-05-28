function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
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

% add bias input to X
X = [ones(rows(X),1) X];

network_matrices = {Theta1, Theta2};
num_layers = length(network_matrices) + 1; % +1 for output layer

p = [];

for i = 1 : m
	% values at activation nodes: i-th element refers to activation nodes in the i-th layer
	A = {};
	A{1} = X(i, :)'; % first activation nodes are the inputs

	% 
	% Forward propagation
	%
	% move through the layers
	for j = 1 : num_layers - 1
		z = cell2mat(network_matrices(j)) * A{j};
		A{j+1} = 1 ./ (1 + exp(-z));
		if j + 1 != num_layers
			A{j+1} = [1; A{j+1}]; % add bias node
		end
	end

	output = A{num_layers}'; % turn the column of output layer into a vector [out_0 ... out_9]
	[~, index] = max(output,[],2); % the column index (i.e. the class) for which the probability to belong to that class is max

	p = [p; index];
end


% =========================================================================


end
