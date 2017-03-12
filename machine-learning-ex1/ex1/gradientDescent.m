function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%disp(X);
%disp("y");
%disp(y);
%disp("theta");
%disp(theta);
%disp(alpha);
%disp(num_iters);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    x = (theta(1)  + (theta(2)*X(:,2))) - y;
    
    forTheta1 = sum(x)/m;
    forTheta2 = sum(((theta(1)  + (theta(2)*X(:,2))) - y).*X(:,2))/m;
    
    theta(1) = theta(1) - alpha*forTheta1;
    theta(2) = theta(2) - alpha*(forTheta2);






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
