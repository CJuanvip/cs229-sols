function [ind, thresh] = find_best_threshold(X, y, p_dist)
    % FIND_BEST_THRESHOLD Finds the best threshold for the given data
    %
    % [ind, thresh] = find_best_threshold(X, y, p_dist) returns a threshold
    % thresh and index ind that gives the best thresholded classifier for the
    % weights p_dist on the training data. That is, the returned index ind
    % and threshold thresh minimize
    %
    % sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
    %
    % OR
    %
    % sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}.
    %
    % We must check both signed directions, as it is possible that the best
    % decision stump (coordinate threshold classifier) is of the form
    % sign(threshold - x_j) rather than sign(x_j - threshold).
    %
    % The data matrix X is of size m-by-n, where m is the training set size
    % and n is the dimension.
    %
    % The solution version uses efficient sorting and data structures to perform
    % this calculation in time O(n m log(m)), where the size of the data matrix
    % X is m-by-n.
    [mm, nn] = size(X);
    best_err = inf;
    ind = 1;
    thresh = 0;
    for jj = 1:nn
        [x_sort, inds] = sort(X(:, jj), 1, ’descend’);
        p_sort = p_dist(inds);
        y_sort = y(inds);
        % We let the thresholds be s_0, s_1, ..., s_{m-1}, where s_k is between
        % x_sort(k-1) and x_sort(k) (so that s_0 > x_sort(1)). Then the empirical
        % error associated with threshold s_k is exactly
        %
        % err(k) = sum_{l = k + 1}^m p_sort(l) * 1(y_sort(l) == 1)
        %        + sum_{l = 1}^k p_sort(l) * 1(y_sort(l) == -1),
        %
        % because this is where the thresholds fall. Then we can sequentially
        % compute
        %
        % err(l) = err(l - 1) - p_sort(l) y_sort(l),
        %
        % where err(0) = p_sort’ * (y_sort == 1).
        %
        % The code below actually performs this calculation with indices shifted by
        % one due to Matlab indexing.
        s = x_sort(1) + 1;
        possible_thresholds = x_sort;
        possible_thresholds = (x_sort + circshift(x_sort, 1)) / 2;
        possible_thresholds(1) = x_sort(1) + 1;
        increments = circshift(p_sort .* y_sort, 1);
        increments(1) = 0;
        emp_errs = ones(mm, 1) * (p_sort’ * (y_sort == 1));
        emp_errs = emp_errs - cumsum(increments);
        [best_low, thresh_ind] = min(emp_errs);
        [best_high, thresh_high] = max(emp_errs);
        best_high = 1 - best_high;
        best_err_j = min(best_high, best_low);
        if (best_high < best_low)
            thresh_ind = thresh_high;
        end
        if (best_err_j < best_err)
            ind = jj;
            thresh = possible_thresholds(thresh_ind);
            best_err = best_err_j;
        end
    end

function [theta, feature_inds, thresholds] = stump_booster(X, y, T)
    % STUMP_BOOSTER Uses boosted decision stumps to train a classifier
    %
    % [theta, feature_inds, thresholds] = stump_booster(X, y, T)
    % performs T rounds of boosted decision stumps to classify the data X,
    % which is an m-by-n matrix of m training examples in dimension n,
    % to match y.
    %
    % The returned parameters are theta, the parameter vector in T dimensions,
    % the feature_inds, which are indices of the features (a T dimensional
    % vector taking values in {1, 2, ..., n}), and thresholds, which are
    % real-valued thresholds. The resulting classifier may be computed on an
    % n-dimensional training example by
    %
    % theta’ * sign(x(feature_inds) - thresholds).
    %
    % The resulting predictions may be computed simultaneously on an
    % n-dimensional dataset, represented as an m-by-n matrix X, by
    %
    % sign(X(:, feature_inds) - repmat(thresholds’, m, 1)) * theta.
    %
    % This is an m-vector of the predicted margins.
    [mm, nn] = size(X);
    p_dist = ones(mm, 1);
    p_dist = p_dist / sum(p_dist);
    theta = [];
    feature_inds = [];
    thresholds = [];
    for iter = 1:T
        [ind, thresh] = find_best_threshold(X, y, p_dist);
        Wplus = p_dist’ * (sign(X(:, ind) - thresh) == y);
        Wminus = p_dist’ * (sign(X(:, ind) - thresh) ~= y);
        theta = [theta; .5 * log(Wplus / Wminus)];
        feature_inds = [feature_inds; ind];
        thresholds = [thresholds; thresh];
        p_dist = exp(-y .* (...
            sign(X(:, feature_inds) - repmat(thresholds’, mm, 1)) * theta));
        fprintf(1, 'Iter %d, empirical risk = %1.4f, empirical error = %1.4f\n', ...
            iter, sum(p_dist), sum(p_dist >= 1));
        p_dist = p_dist / sum(p_dist);
    end

function [theta, feature_inds, thresholds] = random_booster(X, y, T)
    % RANDOM_BOOSTER Uses random thresholds and indices to train a classifier
    %
    % [theta, feature_inds, thresholds] = random_booster(X, y, T)
    % performs T rounds of boosted decision stumps to classify the data X,
    % which is an m-by-n matrix of m training examples in dimension n.
    %
    % The returned parameters are theta, the parameter vector in T dimensions,
    % the feature_inds, which are indices of the features (a T dimensional vector
    % taking values in {1, 2, ..., n}), and thresholds, which are real-valued
    % thresholds. The resulting classifier may be computed on an n-dimensional
    %
    % theta’ * sgn(x(feature_inds) - thresholds).
    %
    [mm, nn] = size(X);
    p_dist = ones(mm, 1);
    p_dist = p_dist / sum(p_dist);
    theta = [];
    feature_inds = [];
    thresholds = [];

    for iter = 1:T
        ind = ceil(rand * nn);
        thresh = X(ceil(rand * mm), ind) + 1e-8 * randn;
        Wplus = p_dist’ * (sign(X(:, ind) - thresh) == y);
        Wminus = p_dist’ * (sign(X(:, ind) - thresh) ~= y);
        theta = [theta; .5 * log(Wplus / Wminus)];
        feature_inds = [feature_inds; ind];
        thresholds = [thresholds; thresh];
        p_dist = exp(-y .* (...
            sign(X(:, feature_inds) - repmat(thresholds’, mm, 1)) * theta));
        fprintf(1, 'Iter %d, empirical risk = %1.4f, empirical error = %1.4f\n', ...
            iter, sum(p_dist), sum(p_dist >= 1));
        p_dist = p_dist / sum(p_dist);
    end

function s = sgn(v)
    s = 2 * (v >= 0) - 1;
