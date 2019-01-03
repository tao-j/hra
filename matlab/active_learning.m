%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Crowd-BT Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [mu, sigma, alpha, beta, accuracy, hist]...
    = active_learning(data, budget, mu, sigma, alpha, beta, theta, rho, trial, para)

    % para sets the parameters of the algorithm
    gamma = getOpt(para,'gamma', 0); % balances exploration-exploitation
    calc_iter = getOpt(para,'calc_iter', 100); 
    anno_threshold = getOpt(para,'anno_threshold', 1e-4);
    verbose = getOpt(para,'verbose', true);
    sel_method = getOpt(para,'sel_method', 'greedy');

    n_data = size(data,1);
    n_obj = length(mu);
    accuracy = zeros(1,budget);
    
    [score, try_result] = init_score(data, mu, sigma, alpha, beta, para);

    hist = struct('seq', zeros(n_obj,1), 'score', ones(n_obj,1));
    candidate = true(n_data,1);
    
    for iter = 1:budget
        
       %% Select the highest score --- greedy algorithm
        if strcmp(sel_method, 'greedy')
            [hist.score(iter)] = max(score);            
            idx = find(score==hist.score(iter));
            if length(idx) == 1
                r = idx;
            else                
                r = idx(randsample(length(idx),1));
            end          
        elseif strcmp(sel_method, 'multinomial')
            r = find(mnrnd(1,score./sum(score)));
        elseif strcmp(sel_method, 'random')
            r = randsample(find(candidate),1);
        end
            
        hist.seq(iter)=r;      
        candidate(r)=false;
        score(r)=0;        
        
       %% Reveal the results i>_k j  and update parameters 
        i = data(r, 2);
        j = data(r, 3);
        k = data(r, 1);
        
        split = rho(k) * theta(i) / (theta(i) + theta(j)) + (1 - rho(k)) * theta(j) / (theta(i) + theta(j));
        if rand > split
            i = data(r,3);
            j = data(r,2);
        end
      
        mu(i) = try_result{r,1}.mu1;
        mu(j) = try_result{r,1}.mu2;
        sigma(i) = try_result{r,1}.sigma1;
        sigma(j) = try_result{r,1}.sigma2;
        
       %% Update the new score and try_result         
        if abs(try_result{r,1}.alpha-alpha(k))<anno_threshold && abs(try_result{r,1}.beta-beta(k)) < anno_threshold
            update_list = find( ((data(:,2)==i) | (data(:,3)==j) | (data(:,2)==j) | (data(:,3)==i)) & candidate);
        else
            update_list = find( (data(:,1) ==k | (data(:,2)==i) | (data(:,3)==j)| (data(:,2)==j) | (data(:,3)==i)) & candidate);
        end
        alpha(k) = try_result{r,1}.alpha;
        beta(k) = try_result{r,1}.beta;
        
        for rr = 1:length(update_list)
            r = update_list(rr);
            i = data(r,2);
            j = data(r,3);
            k = data(r,1);
            [try_result{r,1}.mu1, try_result{r,1}.mu2, try_result{r,1}.sigma1, try_result{r,1}.simga2, try_result{r,1}.alpha,  try_result{r,1}.beta,...
                KL_win_o, KL_win_a, win_prob]=online_update(mu(i), mu(j), sigma(i), sigma(j), alpha(k), beta(k), para);
            [try_result{r,2}.mu1, try_result{r,2}.mu2, try_result{r,2}.sigma1, try_result{r,2}.simga2, try_result{r,2}.alpha,  try_result{r,2}.beta,...
                KL_lose_o, KL_lose_a, lose_prob]=online_update(mu(j), mu(i), sigma(j), sigma(i), alpha(k), beta(k), para);
            score(r) = win_prob*(KL_win_o+gamma*KL_win_a)+lose_prob*(KL_lose_o+gamma*KL_lose_a);
        end
        
       %% Compute and print the Kendall's tau 
        dist = 0;
        for xx = 1:n_obj
            for yy = xx+1:n_obj
                if (mu(xx) - mu(yy))*(theta(xx) - theta(yy)) > 0
                    dist = dist + 1;
                end
            end
        end
        accuracy(iter) = 2 * dist / (n_obj * (n_obj-1));
        
        fprintf('Trial = %d, Iter = %d, Accuracy = %f \n', trial, iter, accuracy(iter));
        
    end
    
end