clear
addpath(genpath('./minFunc_2012'));
addpath(genpath('./minConf'));
max_iter=1200;
res_btl=zeros(10, 10, 12);
res_gbtl=zeros(10, 10, 12);
t=clock;
pps=[8,4,2,1];
for pp=1:4
    for idx=1:max_iter
        data=dlmread(sprintf('./read_data/synthetic%d1/%d.txt',pps(pp), idx-1));
        data = data + 1;
    %     anno_quality=dlmread('./read_data/annotator_info.txt');
    %     anno_quality=anno_quality(:,3);
    %     doc_diff=dlmread('./read_data/doc_info.txt');
        doc_diff=1:10;
        doc_diff=doc_diff'/10;

        n_anno=max(data(:,3));
        n_obj=max(max(data(:,1:2)));

        pair=cell(n_anno,1);
        for i=1:n_anno
            pair{i}=data(data(:,3)==i, 1:2);
        end

        %% set up initial parametmers 
        s_init=zeros(n_obj,1);
        alpha_init=ones(n_anno,1);

        para=struct('reg_0', 1, 'reg_s', 0, 'reg_alpha', 0,  'maxiter', 100, 's0', 0,...
                     'uni_weight', true, 'verbose', false, 'tol', 1e-5);

        opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 300, 'optTol', 1e-5, 'progTol', 1e-7);
        base_s=minFunc(@func_s, s_init, opt_s,  ones(n_anno,1), para, pair);
        base_auc=calc_auc(doc_diff, base_s);
        kendall=corr(doc_diff, base_s, 'type', 'Kendall');
        
        fx=idx-1;
        kk=mod(fx, 12)+1;
        fx=floor(fx/12);
        jj=mod(fx, 10)+1;
        fx=floor(fx/10);
        ii=mod(fx, 10)+1;
        
        res_btl(ii, jj, kk) = kendall;
        
        % base_kendall=calc_kendall(doc_diff, base_s, eps);
        % plot(base_s, doc_diff,  'b*');

        s_init=rand(n_obj,1);
        [s,alpha, obj, iter]=alter(base_s, alpha_init, pair, para);
        auc=calc_auc(doc_diff, s);

        % kendall=calc_kendall(doc_diff, s, eps);
        % plot(1:length(p), doc_diff(sort_idx), '*')
        kendall=corr(doc_diff, s, 'type', 'Kendall');
        res_gbtl(ii, jj, kk) = kendall;
        % plot(s, doc_diff,  'r.');
        et = etime(clock,t);
        eta=et*(max_iter-idx)/idx;
        fprintf('\r%f, %f', eta, et);
    end
    fid = fopen(sprintf('./read_data/%db1.txt', pps(pp)),'w');
    fprintf(fid, 'btl-random-opt\n');
    fclose(fid);
    dlmwrite(sprintf('./read_data/%db1.txt', pps(pp)),mean(res_btl, 3),'-append');
    fid = fopen(sprintf('./read_data/%db1.txt', pps(pp)),'a+');
    fprintf(fid, 'crowdbt-opt\n');
    fclose(fid);
    dlmwrite(sprintf('./read_data/%db1.txt', pps(pp)),mean(res_gbtl, 3),'-append');
end