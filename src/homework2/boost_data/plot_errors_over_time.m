function plot_errors_over_time(Xtrain, ytrain, Xtest, ytest, ...
                               theta, feature_inds, thresholds)
% PLOT_ERRORS_OVER_TIME Plots train and test error from boosting
%
% plot_errors_over_time(Xtrain, ytrain, Xtest, ytest, theta, ...
%                       feature_inds, thresholds)
%
% Plots the training and testing error of a decision-stump based boosting
% algorithm over iterations of the boosting algorithm.

num_thresholds = length(thresholds);

train_errors = zeros(num_thresholds, 1);
test_errors = zeros(num_thresholds, 1);

mtrain = size(Xtrain, 1);
mtest = size(Xtest, 1);

% Predicted margins for train and test
train_predictions = zeros(mtrain, 1);
test_predictions = zeros(mtest, 1);

% Iteratively compute the margin predicted by the thresholded classifier,
% updating both test and training predictions.
for iter = 1:num_thresholds
  train_predictions = train_predictions + ...
      sign(Xtrain(:, feature_inds(iter)) - thresholds(iter)) * theta(iter);
  test_predictions = test_predictions + ...
      sign(Xtest(:, feature_inds(iter)) - thresholds(iter)) * theta(iter);
  train_errors(iter) = sum((ytrain .* train_predictions) <= 0) / mtrain;
  test_errors(iter) = sum((ytest .* test_predictions) <= 0) / mtest;
end

figure;
h_train = plot(train_errors);
set(h_train, 'linewidth', 2);
set(h_train, 'color', [0, 0, .9]);
hold on;
h_test = plot(test_errors);
set(h_test, 'linewidth', 2);
set(h_test, 'color', [0, 0, 0]);
set(gca, 'fontsize', 18);
legend([h_train, h_test], {'Train error rate', 'Test error rate'});
ylabel('Error rate');
xlabel('Iterations');
grid on;
