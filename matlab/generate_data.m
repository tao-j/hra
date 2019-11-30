function [pair, data]=generate_data(s_gt, gt_gamma, n_obj, n_anno, alpha, dist_flag)
    GUMBEL = 'g';
    NORMAL = 'n';
    
%     comb_idx = combnk(1:n_obj, 2); % no repeated comparison, compare 1 time
    comb_idx = cartprod(1:n_obj, 1:n_obj); % at most compare 2 times
    comb_idx = comb_idx(comb_idx(:, 1) ~= comb_idx(:, 2), :); % discard compare to itself
    comb_s = s_gt(comb_idx); % create tuples of all possible pairwise comparisons

    n_comb = size(comb_idx, 1); % number of combinations
    pair = cell(n_anno, 1);
    for w = 1:n_anno
        this_gamma = gt_gamma(w);

        if this_gamma > 0
            flip_flag = false;
        else
            flip_flag = true;
            this_gamma = -this_gamma;
        end

        % Gumble Distribution f or noise
        if dist_flag == GUMBEL
            noise = evrnd(0.5772 * 1. / this_gamma, 1. / this_gamma, size(comb_idx));

        % Normal Distribution for noise
        elseif dist_flag == NORMAL
            noise = normrnd(0., 1. / this_gamma, size(comb_idx));
        else
            % not implemented error
            assert(false);
        end
        obsv = comb_s + noise;

        flip_idx = obsv(:,1) < obsv(:,2);
        if flip_flag % flip comparisions if gamma is negatie
            flip_idx = ~flip_idx;
        end

        this_comb = comb_idx;
        this_comb(flip_idx, :) = [this_comb(flip_idx, 2) this_comb(flip_idx, 1)];
        % discard data according to ratio
        this_comb = this_comb(rand(n_comb, 1) <= alpha, :);

        pair{w} = this_comb;
        n_this_comb = size(this_comb, 1);
        if w == 1
            data(1:n_this_comb, 1) = w;
            data(1:n_this_comb, 2:3) = this_comb;
        else
            data(end+1:end+n_this_comb, 1) = w;
            data(end+1:end+n_this_comb, 2:3) = this_comb;
        end
    end
end
