%% Read data
clear;
addpath(genpath('./minFunc_2012'));
addpath(genpath('./minConf'));
data=dlmread('./data/readinglevel/all_pair.txt');
% anno_quality=dlmread('./data/readinglevel/annotator_info.txt');
% anno_quality=anno_quality(:,3);
doc_diff=dlmread('./data/readinglevel/doc_info.txt');
doc_diff=doc_diff(:,2);

n_anno=max(data(:,1));
n_obj=max(max(data(:,2:3)));

pair=cell(n_anno,1);
for i=1:n_anno
    pair{i}=data(data(:,1)==i, 2:3);
end

exps=20;
res = cell(exps, 1);
legends_cell = cell(exps, 1);
markers = {'+','o','*','.','x','s','d','^','v','>','<','p','h','+','o','*','.','x','s','d','^','v','>','<','p','h'};
res_idx = 1;


%% set up initial parametmers 
sc=1;
para=struct('reg_0', 1., 'reg_s', 0, 'reg_alpha', 0,  'maxiter', 600, 's0', 0,...
             'uni_weight', true, 'verbose', true, 'tol', 1e-5);
% para.algo='CrowdBT';
% para.algo='HRA-G';
% para.algo='HRA-N';
% para.algo='HRA-E';
% para.opt_method='a->s+GD';
% para.opt_method='s->a+newton+crowdbt';
para.lr=5*10e-4;
para.alpha_rate = 0.25;

%% Random BTL-MLE
name='Random BTL-MLE';
para.algo='HRA-G';
s_init=rand(n_obj,1);
alpha_init=ones(n_anno,1);
opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 300, 'optTol', 1e-5, 'progTol', 1e-7);
random_btl_mle_s=minFunc(@func_s, s_init*sc, opt_s, ones(n_anno,1)/sc, para, pair);
% obj=ones(1000,1);
% s = s_init;
% lr = 10e-1;
% for iter=1:1000
%     [obj(iter), g_s] = func_s(s, ones(n_anno, 1)*50, para, pair);
%     s = s - lr*g_s;
%     base_auc=calc_auc(doc_diff, s)
% end
% base_s = s;
auc=calc_auc(doc_diff, random_btl_mle_s);
kendall=corr(doc_diff, random_btl_mle_s, 'type', 'Kendall');
hold off;
plot(1:60,ones(60, 1) * func_s(random_btl_mle_s, ones(n_anno, 1)/sc, para, pair),markers{res_idx});
hold all;
legend_cell{res_idx}=name;
res{res_idx}={name, auc, kendall};
res_idx=res_idx+1;


% base_kendall=calc_kendall(doc_diff, base_s, eps);
% plot(base_s, doc_diff,  'b*'); 

%% BTL-MLE + HRA-G
name='BTL-MLE + HRA-G';
fprintf([name '\n']);
para.algo='HRA-G';
para.opt_method='a->s+GD';
[s,alpha, obj, iter]=alter(random_btl_mle_s*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx}={name, auc, kendall};
legend_cell{res_idx}=name;
res_idx=res_idx+1;
plot(obj(1:10:600), markers{res_idx})

%% BTL-MLE + HRA-N
name='BTL-MLE + HRA-N';
fprintf([name '\n']);
para.algo='HRA-N';
para.opt_method='a->s+GD';
[s,alpha, obj, iter]=alter(random_btl_mle_s*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx}={name, auc, kendall};
plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
res_idx=res_idx+1;

%% BTL-MLE + HRA-E
name='BTL-MLE + HRA-E';
fprintf([name '\n']);
para.algo='HRA-E';
para.opt_method='a->s+GD';
[s,alpha, obj, iter]=alter(random_btl_mle_s*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx}={name, auc, kendall};
plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
res_idx=res_idx+1;

%% Ones + CrowdBT
name='Ones CrowdBT';
fprintf([name '\n']);
para.algo='CrowdBT';
para.opt_method='s->a+newton+crowdbt';
s_init=ones(n_obj,1);
alpha_init=ones(n_anno,1);
[s, alpha, obj, iter]=alter(s_init*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx}={name, auc, kendall};
plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
res_idx=res_idx+1;

ones_crowd_bt_s=s;

%% Ones + CrowdTCV
name='Ones CrowdTCV';
fprintf([name '\n']);
para.algo='CrowdTCV';
para.opt_method='s->a+newton+crowdbt';
s_init=ones(n_obj,1);
alpha_init=ones(n_anno,1);
[s, alpha, obj, iter]=alter(s_init*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx}={name, auc, kendall};
plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
res_idx=res_idx+1;

ones_crowd_bt_s=s;
% 
% 
% %% staring random init
% s_init=rand(n_obj,1);
% alpha_init=ones(n_anno,1);
% %% Random CrowdBT
% name='Random CrowdBT';
% fprintf([name '\n']);
% para.algo='CrowdBT';
% para.opt_method='s->a+newton+crowdbt';
% opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 300, 'optTol', 1e-5, 'progTol', 1e-7);
% [s, alpha, obj, iter]=alter(s_init*sc, (alpha_init/sc), pair, para);
% 
% auc=calc_auc(doc_diff, s);
% kendall=corr(doc_diff, s, 'type', 'Kendall');
% res{res_idx}={name, auc, kendall};
% plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
% res_idx=res_idx+1;
% 
% random_crowd_bt_s=s;
% 
% % base_kendall=calc_kendall(doc_diff, base_s, eps);
% % plot(base_s, doc_diff,  'b*'); 
% 
% %% Random + HRA-G
% name='Random + HRA-G s->a';
% fprintf([name '\n']);
% para.algo='HRA-G';
% para.opt_method='s->a+GD';
% [s,alpha, obj, iter]=alter(s_init*sc, (alpha_init/sc), pair, para);
% 
% auc=calc_auc(doc_diff, s);
% kendall=corr(doc_diff, s, 'type', 'Kendall');
% res{res_idx}={name, auc, kendall};
% plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
% res_idx=res_idx+1;
% 
% %% Random + HRA-N
% name='Random + HRA-N s->a';
% fprintf([name '\n']);
% para.algo='HRA-N';
% para.opt_method='s->a+GD';
% [s,alpha, obj, iter]=alter(s_init*sc, (alpha_init/sc), pair, para);
% 
% auc=calc_auc(doc_diff, s);
% kendall=corr(doc_diff, s, 'type', 'Kendall');
% res{res_idx}={name, auc, kendall};
% plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
% res_idx=res_idx+1;
% 
% %% Random + HRA-E
% name='Random + HRA-E s->a';
% fprintf([name '\n']);
% para.algo='HRA-E';
% para.opt_method='s->a+GD';
% [s,alpha, obj, iter]=alter(s_init*sc, (alpha_init/sc), pair, para);
% 
% auc=calc_auc(doc_diff, s);
% kendall=corr(doc_diff, s, 'type', 'Kendall');
% res{res_idx}={name, auc, kendall};
% plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
% res_idx=res_idx+1;

% %% CrowdBT + HRA-G
% name='CrowdBT + HRA-G';
% fprintf([name '\n']);
% para.algo='HRA-G';
% para.opt_method='a->s+GD';
% [s,alpha, obj, iter]=alter(random_crowd_bt_s*sc, (alpha_init/sc), pair, para);
% 
% auc=calc_auc(doc_diff, s);
% kendall=corr(doc_diff, s, 'type', 'Kendall');
% res{res_idx}={name, auc, kendall};
% plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
% res_idx=res_idx+1;
% 
% %% CrowdBT + HRA-N
% name='CrowdBT + HRA-N';
% fprintf([name '\n']);
% para.algo='HRA-N';
% para.opt_method='a->s+GD';
% [s,alpha, obj, iter]=alter(random_crowd_bt_s*sc, (alpha_init/sc), pair, para);
% 
% auc=calc_auc(doc_diff, s);
% kendall=corr(doc_diff, s, 'type', 'Kendall');
% res{res_idx}={name, auc, kendall};
% plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
% res_idx=res_idx+1;
% 
% %% CrowdBT + HRA-E
% name='CrowdBT + HRA-E';
% fprintf([name '\n']);
% para.algo='HRA-E';
% para.opt_method='a->s+GD';
% [s,alpha, obj, iter]=alter(random_crowd_bt_s*sc, (alpha_init/sc), pair, para);
% 
% auc=calc_auc(doc_diff, s);
% kendall=corr(doc_diff, s, 'type', 'Kendall');
% res{res_idx}={name, auc, kendall};
% plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
% res_idx=res_idx+1;
% 
% alpha_init=ones(n_anno, 1);
%% Ones + HRA-G
name='Ones + HRA-G s->a';
fprintf([name '\n']);
para.algo='HRA-G';
para.opt_method='s->a+GD';
[s,alpha, obj, iter]=alter(ones(n_obj, 1)*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx}={name, auc, kendall};
plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
res_idx=res_idx+1;

%% Ones + HRA-N
name='Ones + HRA-N s->a';
fprintf([name '\n']);
para.algo='HRA-N';
para.opt_method='s->a+GD';
[s,alpha, obj, iter]=alter(ones(n_obj, 1)*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx}={name, auc, kendall};
plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
res_idx=res_idx+1;

%% Ones + HRA-E
name='Ones + HRA-E s->a';
fprintf([name '\n']);
para.algo='HRA-E';
para.opt_method='s->a+GD';
[s,alpha, obj, iter]=alter(ones(n_obj, 1)*sc, (alpha_init/sc), pair, para);

auc=calc_auc(doc_diff, s);
kendall=corr(doc_diff, s, 'type', 'Kendall');
res{res_idx}={name, auc, kendall};
if auc < 0.0001
    plot(10000, markers{res_idx});legend_cell{res_idx}=name;
else
    plot(obj(1:10:600), markers{res_idx});legend_cell{res_idx}=name;
end
res_idx=res_idx+1;

%% Final clean up
res=res(1:res_idx-1);
legend_cell=legend_cell(1:res_idx-1);
legend(legend_cell);
% m = s * ones(1, n_anno);
% [~, idx] = sort(m);
% spread = (m(idx(n_anno/3)) - m(idx(n_anno/3*2))) .* alpha;
% mean_ratio = mean(spread)

for i=1:res_idx-1
   fprintf('%s,%f,%f\n', res{i}{1,1},res{i}{1,2},res{i}{1,3}); 
end