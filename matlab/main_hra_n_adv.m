%% Basic inputs
clear;
addpath(genpath('./minFunc_2012'));
addpath(genpath('./minConf'));
rng(2333);

n_algo = 3;
n_anno_good = 3;
n_anno_bad = 6;
n_anno = n_anno_good + n_anno_bad;  % number of workers
gamma_goods = [2.5 5 10];
gamma_bads = [0.25 1 2.5];

n_obj = 10;   % number of items
trials = 100;  % number of independent trials to run

%% Test

record_var = zeros(n_algo, 9); % record of ranking accuracy
record_mean = zeros(n_algo, 9); % record of ranking accuracy
alpha = 0.8;
s = linspace(1./n_obj, 1., n_obj);
doc_diff=s';
% all_comb = combnk(1:n_obj, 2);
all_comb = cartprod(1:n_obj, 1:n_obj);
all_comb = all_comb(all_comb(:, 1) ~= all_comb(:, 2), :);
s_comb = s(all_comb);

[m,~] = size(all_comb);

param_idx = 1;
for gg = 1:3
for gb = 1:3
    gamma_good = gamma_goods(gg);
    gamma_bad = gamma_bads(gb);
    gt_gamma = [repmat(gamma_good, 1, n_anno_good.*2/3) repmat(gamma_bad, 1, n_anno_bad.*2/3) repmat(-gamma_good, 1, n_anno_good.*1/3) repmat(-gamma_bad, 1, n_anno_bad.*1/3)]; % generate workers

this_record = zeros(n_algo, trials);
for tr = 1:trials

    % s = drchrnd(ones(1,n_obj),1); % generate items

    % create data
    data = zeros(n_anno*m,3);
    for w = 1:n_anno
        this_gamma = gt_gamma(w);
        
        if this_gamma > 0
        noise = normrnd(0., 1. / this_gamma, size(all_comb));
        obsv = s_comb + noise;
        flip_idx = obsv(:,1) < obsv(:,2);
        this_comb = all_comb;
        this_comb(flip_idx, :) = [this_comb(flip_idx, 2) this_comb(flip_idx, 1)];
        else
        noise = normrnd(0., 1. / -this_gamma, size(all_comb));
        obsv = s_comb + noise;
        flip_idx = obsv(:,1) > obsv(:,2);
        this_comb = all_comb;
        this_comb(flip_idx, :) = [this_comb(flip_idx, 2) this_comb(flip_idx, 1)];
        end

        before = (w-1)*m+1;
        after = w*m;
        data(before:after,1) = w;
        data(before:after,2:3) = this_comb;
    end
    
    data=data(rand(m*n_anno, 1) <= alpha, :);

    pair=cell(n_anno,1);
    for i=1:n_anno
        pair{i}=data(data(:,1)==i, 2:3);
    end
    
    para=struct('reg_0', 0., 'reg_s', 0, 'reg_alpha', 0,  'maxiter', 600, 's0', 0,...
             'uni_weight', true, 'verbose', false, 'tol', 1e-5);
    % para.algo='CrowdBT';
    % para.algo='HRA-G';
    % para.algo='HRA-N';
    % para.algo='HRA-E';
    % para.opt_method='a->s+GD';
    % para.opt_method='s->a+newton+crowdbt';
    para.lr=5*10e-4;
    
    % initial parameters
%     s_init = ones(n_obj,1);
    s_init=rand(n_obj,1);
    gamma_init = ones(n_anno,1);
    sc = 1;
    
    %% Ones BTL-MLE
    para.algo='HRA-N';
    opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 300, 'optTol', 1e-5, 'progTol', 1e-7);
    
    random_btl_mle_s=minFunc(@func_s, s_init*sc, opt_s, gamma_init/sc, para, pair);
    kendall=corr(doc_diff, random_btl_mle_s, 'type', 'Kendall');
    this_record(1,tr) = kendall;
    
    %% Ones CrowdBT
    name='Random CrowdBT';
    fprintf([name '\n']);
    para.algo='CrowdBT';
    para.opt_method='s->a+newton+crowdbt';
    [s, ~, ~, ~]=alter(s_init*sc, (gamma_init/sc), pair, para);
    
    kendall=corr(doc_diff, s, 'type', 'Kendall');
    this_record(2, tr) = kendall;
    
    %% Ones + HRA-N
    name='Ones + HRA-N';
    fprintf([name '\n']);
    para.algo='HRA-N';
    para.opt_method='a->s+GD';
    [s, gamma, obj, iter]=alter(s_init*sc, (gamma_init/sc), pair, para);

    kendall=corr(doc_diff, s, 'type', 'Kendall');
    this_record(3, tr) = kendall;
    


end
    record_mean(:, param_idx) = mean(this_record, 2);
    record_var(:, param_idx) = std(this_record, 0, 2);
    param_idx = param_idx + 1;
end
end


