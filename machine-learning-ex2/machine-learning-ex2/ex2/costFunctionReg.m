function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
g = sigmoid(X*theta);
J = -((log(g))'*y+(log(1-g))'*(1-y))/m+(lambda*(theta(2:end)'*theta(2:end)))/(2*m);
grad(1) = X(:,1)'*(g-y)/m;
grad(2:end) = X(:,2:end)'*((g-y)/m)+lambda*theta(2:end)/m;





% =============================================================

end
