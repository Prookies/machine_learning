function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
accuracies = zeros(length(C_val), length(sigma_val));

for i = 1:length(C_val)
    for j = 1:length(sigma_val)
        model = svmTrain(X, y, C_val(i), @(x1, x2)gaussianKernel(x1,x2,sigma_val(j)));
        predictions = svmPredict(model, Xval);
        accuracies(i, j) = mean(double(predictions ~= yval));
        fprintf('C = %f, sigma = %f    ',C_val(i),sigma_val(j));
        fprintf('accuracies(%d,%d) = %f\n',i,j,accuracies(i,j));
    end
end
accuracies
error=1;
for i = 1:length(C_val)
    for j = 1:length(sigma_val)
        if accuracies(i,j) < error
            C = C_val(i);
            sigma = sigma_val(j);
            error = accuracies(i,j);
        end
    end
end
        





% =========================================================================

end
