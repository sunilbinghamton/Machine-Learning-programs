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


hypo = sigmoid(X * theta);
a = log(hypo);
b = log(1 - hypo);

errorcost = (-1 * (y .* a)) - ( (1 - y ) .* b);
regfactor = (lambda/(2*m)) * ( sum(theta.^2) - theta(1)^2 ) ;
J = sum(errorcost) *( 1/m) + regfactor;

gradfactor = [0; ((lambda/m) * theta(2:end)) ];
grad = ( (1/ m) .* ( X' * (hypo - y)) ) + gradfactor;



% =============================================================

end
