%% Read data
clear;
addpath(genpath('./minFunc_2012'));
addpath(genpath('./minConf'));
data=dlmread('./read_data/all_pair.txt');
anno_quality=dlmread('./read_data/annotator_info.txt');
anno_quality=anno_quality(:,3);
doc_diff=dlmread('./read_data/doc_info.txt');
doc_diff=doc_diff(:,2);

n_anno=max(data(:,1));
n_obj=max(max(data(:,2:3)));

pair=cell(n_anno,1);
for i=1:n_anno
    pair{i}=data(data(:,1)==i, 2:3);
end


%% set up initial parametmers 
s_init=rand(n_obj,1);
alpha_init=ones(n_anno,1);
sc=2.;
para=struct('reg_0', 1., 'reg_s', 0, 'reg_alpha', 0,  'maxiter', 300, 's0', 0,...
             'uni_weight', true, 'verbose', true, 'tol', 1e-5);
para.algo='CrowdBT';
para.algo='HRA-G';
para.algo='HRA-N';
%para.algo='HRA-E';
para.opt_method='s->a+GD';
%para.lr=0.75*10e-4;

%% baseline
opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 300, 'optTol', 1e-5, 'progTol', 1e-7);
base_s=minFunc(@func_s, s_init*sc, opt_s, (alpha_init/sc), para, pair);
% obj=ones(1000,1);
% s = s_init;
% lr = 10e-1;
% for iter=1:1000
%     [obj(iter), g_s] = func_s(s, ones(n_anno, 1)*50, para, pair);
%     s = s - lr*g_s;
%     base_auc=calc_auc(doc_diff, s)
% end
% base_s = s;

base_auc=calc_auc(doc_diff, base_s)
kendall=corr(doc_diff, base_s, 'type', 'Kendall')

% base_kendall=calc_kendall(doc_diff, base_s, eps);
% plot(base_s, doc_diff,  'b*');

%% HRA
[s,alpha, obj, iter]=alter(base_s*sc, (alpha_init/sc), pair, para);
alt_opt_auc=calc_auc(doc_diff, s)

% kendall=calc_kendall(doc_diff, s, eps);
% plot(1:length(p), doc_diff(sort_idx), '*')
kendall=corr(doc_diff, s, 'type', 'Kendall')
% plot(s, doc_diff,  'r.');
plot(obj)

m = s * ones(1, n_anno);
[~, idx] = sort(m);
spread = (m(idx(n_anno/3)) - m(idx(n_anno/3*2))) .* alpha;
mean_ratio = mean(spread)