close all;

M = csvread('boosting-train.csv');
ytrain = M(:, 1);
Xtrain = M(:, 2:end);

M = csvread('boosting-test.csv');
ytest = M(:, 1);
Xtest = M(:, 2:end);

%% Now do training of the boosters %%

[theta_rnd, feature_inds_rnd, thresholds_rnd] = ...
    random_booster(Xtrain, ytrain, 200);

[theta, feature_inds, thresholds] = stump_booster(Xtrain, ytrain, 200);

plot_errors_over_time(Xtrain, ytrain, Xtest, ytest, ...
                      theta_rnd, feature_inds_rnd, thresholds_rnd);
title('Random boosting error');

plot_errors_over_time(Xtrain, ytrain, Xtest, ytest, ...
                      theta, feature_inds, thresholds);
title('Boosted decision stumps error');
