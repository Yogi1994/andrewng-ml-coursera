function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% method one easily understandable
temp = 0;
for i = 1:m
  h = theta(1) + theta(2)*X(i,2);
  %disp(h-y(i));
  temp = temp + (h - y(i))^2;
end

J = temp/(2*m);

% Method 2 :P easy code and thats correct
%costs = (X * theta - y) .^ 2;
%J = sum(costs) / (2 * m);
% =========================================================================

end
