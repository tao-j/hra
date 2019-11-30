function []=main(dist_flag, adv_flag)
addpath(genpath('./minFunc_2012'));
addpath(genpath('./minConf'));
rng(2333);
GUMBEL = 'g';
NORMAL = 'n';

%% configuration
% dist_flag = GUMBEL; % Gumbel for noise distribution
% dist_flag = NORMAL; % Normal for ..
% adv_flag = false; % adversarial case (1/3 annotators flip their decision)
% adv_flag = true;

lr = 15*10e-4;
ar = 0.15;
reg_0 = 0.;

n_algo = 3;   % number of algorithms to test
n_obj = 20;   % number of items
trials = 100;  % number of independent trials to run
n_anno_good = 3;
n_anno_bad = 6;
n_anno = n_anno_good + n_anno_bad;  % number of workers
gamma_goods = [2.5 5 10]; % gamma_A
gamma_bads = [0.25 1 2.5]; % gamma_B
s_gt = linspace(1./n_obj, 1., n_obj)'; % generate ground truth s

% write result to file....
if adv_flag
    adv_str = "-adv";
else
    adv_str = "";
end
out_f = fopen(sprintf("./data/simulation/dump_%s%s.txt", dist_flag, adv_str), 'w');

[gg_s, ~] = size(gamma_goods);
[gb_s, ~] = size(gamma_bads);
n_record = gg_s * gb_s;

% alpha = 0.2; % portion of all possible pairs compared
for alpha = [0.2, 0.4, 0.6, 0.8]
% for alpha = [0.8]

    record_std = zeros(n_algo, n_record); % record of ranking accuracy, 9 settings for betas
    record_mean = zeros(n_algo, n_record);

    comb_idx = combnk(1:n_obj, 2); % no repeated comparison, compare 1 time
%     comb_idx = cartprod(1:n_obj, 1:n_obj); % at most compare 2 times
    comb_idx = comb_idx(comb_idx(:, 1) ~= comb_idx(:, 2), :); % discard compare to itself
    comb_s = s_gt(comb_idx); % create tuples of all possible pairwise comparisons

    n_comb = size(comb_idx, 1); % number of combinations

    %% do all gamma combinations
    for gg = 1:3
        for gb = 1:3
            gamma_good = gamma_goods(gg);
            gamma_bad = gamma_bads(gb);
            % generate workers
            if ~adv_flag
                gt_gamma = [repmat(gamma_good, 1, n_anno_good) ...
                    repmat(gamma_bad, 1, n_anno_bad)];
            else
                % make 1/3 of annotators to be adversarial
                gt_gamma = [repmat(gamma_good, 1, n_anno_good * 2/3) ...
                    repmat(gamma_bad, 1, n_anno_bad * 2/3) ...
                    repmat(-gamma_good, 1, n_anno_good * 1/3) ...
                    repmat(-gamma_bad, 1, n_anno_bad * 1/3)];
            end

            this_record = zeros(n_algo, trials);
            for trial_idx = 1:trials
                % s = drchrnd(ones(1,n_obj),1); % generate new item for each trial
                fprintf("tr%d gg%d gb%d\n", trial_idx, gg, gb);
                %% generate data
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
                end

                para = struct('reg_0', reg_0, 'reg_s', 0, 'reg_alpha', 0,  'maxiter', 600, 's0', 0,...
                         'uni_weight', true, 'verbose', true, 'tol', 1e-5);
                para.lr=lr;
                para.alpha_rate=ar;
                %% initial parameters
                s_init = ones(n_obj,1);
                % s_init = rand(n_obj,1);
                gamma_init = ones(n_anno,1);
                eta_init = ones(n_anno,1);
                sc = 1;

                %% GUMBEL
                if dist_flag == GUMBEL
                    %% Random BTL-MLE
                    para.algo='HRA-G';
                    opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 300, 'optTol', 1e-5, 'progTol', 1e-7);
                    s_random_btl_mle=minFunc(@func_s, s_init*sc, opt_s, gamma_init/sc, para, pair);

                    kendall=corr(s_gt, s_random_btl_mle, 'type', 'Kendall');
                    this_record(1, trial_idx) = kendall;

                    %% Ones + CrowdBT
                    name='Ones CrowdBT';
                    fprintf([name '\n']);
                    para.algo='CrowdBT';
                    para.opt_method='s->a+newton+crowdbt';
                    [s_crowdbt, gamma_crowdbt, obj_crowdbt, iter_crowdbt] = ...
                        alter(s_init, eta_init, pair, para);

                    kendall=corr(s_gt, s_crowdbt, 'type', 'Kendall');
                    this_record(2, trial_idx) = kendall;

                    %% Ones + HRA-G
                    name='Ones HRA-G';
                    fprintf([name '\n']);
                    para.algo='HRA-G';
                    para.opt_method='a->s+GD';
                    [s_hra_g, gamma_hra_g, obj_hra_g, iter_hra_g] = ...
                        alter(s_init*sc, (gamma_init/sc), pair, para);

                    kendall=corr(s_gt, s_hra_g, 'type', 'Kendall');
                    this_record(3, trial_idx) = kendall;
                

                %% NORMAL
                elseif dist_flag == NORMAL
                    %% Random BTL-MLE
                    para.algo='HRA-N';
                    opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 300, 'optTol', 1e-5, 'progTol', 1e-7);
                    s_random_btl_mle=minFunc(@func_s, s_init*sc, opt_s, gamma_init/sc, para, pair);

                    kendall=corr(s_gt, s_random_btl_mle, 'type', 'Kendall');
                    this_record(1, trial_idx) = kendall;
                    %% Ones + CrowdTCV
                    name='Ones CrowdTCV';
                    fprintf([name '\n']);
                    para.algo='CrowdTCV';
                    para.opt_method='s->a+newton+crowdbt';
                    % para.opt_method='a->s+GD';
                    [s_crowdtcv, gamma_crowdtcv, obj_crowdtcv, iter_crowd_tcv] = ...
                        alter(s_init, eta_init, pair, para);

                    kendall=corr(s_gt, s_crowdtcv, 'type', 'Kendall');
                    this_record(2, trial_idx) = kendall;

                    %% Ones + HRA-N
                    name='Ones HRA-N';
                    fprintf([name '\n']);
                    para.algo='HRA-N';
                    para.opt_method='a->s+GD';
                    [s_hra_n, gamma_hra_n, obj_hra_n, iter_hra_n] = ...
                        alter(s_gt*sc, (gamma_init/sc), pair, para);

                    kendall=corr(s_gt, s_hra_n, 'type', 'Kendall');
                    this_record(3, trial_idx) = kendall;
                else
                    % not implemented error
                    assert(false);
                end
            end
            gamma_idx = (gg - 1)*3 + gb;
            record_mean(:, gamma_idx) = mean(this_record, 2);
            record_std(:, gamma_idx) = std(this_record, 0, 2);
        end
    end
    % for each alpha write reresult to disk
    % out_f opened before
    fprintf(out_f, "alpha = %s\n", num2str(alpha));

    dump_mat(out_f, record_mean);
    dump_mat(out_f, record_std);

end % alpha
end % function

function []=dump_mat(out_f, record)
    [mm, nn] = size(record);
    for i=1:mm
        for j=1:nn
            fprintf(out_f, "%f ", record(i, j));
        end
        fprintf(out_f, "\n");
    end
    fprintf(out_f, "\n");
end
