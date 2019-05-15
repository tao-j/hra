%% Basic inputs
clear;
close all;
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
trials = 1000;  % number of independent trials to run

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

% lr_s = [0.5, 1, 5, 10, 20, 50];
% lr_s = [0.5, 1, 5, 10, 20];
% lr_s = [1, 2, 4, 4.5, 7];
% arates = [0.5, 1, 5, 10, 20, 50];
% arates = [20, 37.5, 120, 75, 200];
% markers = {'-x','-+','-s','.','x','s','d','^','v','>','<','p','h','+','o','*','.','x','s','d','^','v','>','<','p','h'};

% for lridx=3:3
% for aridx=3:3

res_idx = 1;
param_idx = 1;
% gi = [2 1; 2 2; 2 3];
% for gii = 3:3
for ggi = 1:3
for gbi = 1:3
%     ggi = gi(gii, 1);
%     gbi = gi(gii, 2);
    gamma_good = gamma_goods(ggi);
    gamma_bad = gamma_bads(gbi);
    gt_gamma = [repmat(gamma_good, 1, n_anno_good) repmat(gamma_bad, 1, n_anno_bad)]; % generate workers

this_record = zeros(n_algo, trials);
for tr = 1:trials

    % s = drchrnd(ones(1,n_obj),1); % generate items

    % create data
    data = zeros(n_anno*m,3);
    for w = 1:n_anno
        this_gamma = gt_gamma(w);
        noise = evrnd(0.5772 * 1. / this_gamma, 1. / this_gamma, size(all_comb));
%         noise = normrnd(0., 1. / this_gamma, size(all_comb));
        obsv = s_comb + noise;
        flip_idx = obsv(:,1) < obsv(:,2);
        this_comb = all_comb;
        this_comb(flip_idx, :) = [this_comb(flip_idx, 2) this_comb(flip_idx, 1)];

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
    para.lr=10*10e-4;
    para.alpha_rate = 1.;
%     para.alpha_rate=arates(aridx);
    
    % initial parameters
%     s_init = ones(n_obj,1);
    s_init=ones(n_obj,1);
    gamma_init = ones(n_anno,1);
    sc = 1;
    
    %% Ones BTL-MLE
    para.algo='HRA-G';
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
    
    %% Ones + HRA-G
    name='Ones + HRA-G';
    fprintf([name '\n']);
    para.algo='HRA-G';
    para.opt_method='s->a+GD';
    [s, gamma, obj, iter, s_lst, gamma_lst]=alter(s_init*sc, (gamma_init/sc), pair, para);
%     s_lst = gamma_lst(1, :) ./ gt_gamma(1) .* s_lst;
%     gamma_lst = gt_gamma(1) ./ gamma_lst(1, :) .* gamma_lst;
    
    %% print
%     nn = s_lst-doc_diff+mean(doc_diff);
%     nrm_s = sqrt(sum(nn .* nn, 1))';
%     
%     gg = gamma_lst-gt_gamma';
%     nrm_g = sqrt(sum(gg .* gg, 1))';
%     
%     export_val = [nrm_s nrm_g obj./250];
    
    kendall=corr(doc_diff, s, 'type', 'Kendall');
    this_record(3, tr) = kendall;
    
%     subplot(1,3,1);
%     hold all;
%     plot(1:49, nrm_s(2:50), 1:49, nrm_g(2:50));
%     subplot(1,2,1); 
%     hold all;
%     plot(1:30, nrm_g(1:30), markers{res_idx});
%     ylabel('$||\boldmath{\gamma}^{(t)} - \boldmath{\gamma^*}||$','Interpreter','latex') 
%     xlabel('Number of iteration (t)','Interpreter','latex')
%     legend('$\gamma_A=2.5, \gamma_B=0.25$','$\gamma_A=2.5, \gamma_B=1$', '$\gamma_A=2.5, \gamma_B=2.5$','Interpreter','latex')
%     subplot(1,2,2);
%     hold all;
%     plot(1:30, nrm_s(1:30), markers{res_idx});
%     ylabel('$||\boldmath{s}^{(t)} - \boldmath{s^*}||$','Interpreter','latex') 
%     xlabel('Number of iteration (t)','Interpreter','latex') 
%     saveas(gcf, sprintf('fig_g/ar%dlr%dgg%dgb%d-g-5-5-set3full.png', aridx, lridx, ggi, gbi));
%     legend('$\gamma_A=2.5, \gamma_B=0.25$','$\gamma_A=2.5, \gamma_B=1$', '$\gamma_A=2.5, \gamma_B=2.5$','Interpreter','latex')
%     
    res_idx = res_idx + 1;
end
    record_mean(:, param_idx) = mean(this_record, 2);
    record_var(:, param_idx) = std(this_record, 0, 2);
    param_idx = param_idx + 1;
end
end

% end
% end
