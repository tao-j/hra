function [obj,grad]=func_s(s, alpha, para, pair)

    s0=getOpt(para, 's0', 0);
    reg_0=getOpt(para, 'reg_0', 0);
    reg_s=getOpt(para, 'reg_s', 0);
    reg_alpha=getOpt(para, 'reg_alpha', 0);
    uni_weight=getOpt(para, 'uni_weight', true);
    
    p=exp(s);
    p0=exp(s0);
    obj=-reg_0*(sum(log(p0./(p0+p)))+sum(log(p./(p0+p))));
    grad=2*reg_0*(p./(p0+p))-reg_0;

    for k=1:length(pair)
   
        if (uni_weight)
            s_k=1;
        else
            s_k=size(pair{k},1);
        end

        %% CrowdBT
%         obj=obj-sum(log((alpha(k)*p(pair{k}(:,1))+(1-alpha(k))*p(pair{k}(:,2))))-...
%             log(p(pair{k}(:,1))+p(pair{k}(:,2))))/s_k;
%         for idx=1:size(pair{k},1)
%             winner=pair{k}(idx,1);
%             loser=pair{k}(idx,2);
%             v=(p(winner)/(p(winner)+p(loser))...
%                 -alpha(k)*p(winner)/(alpha(k)*p(winner)+(1-alpha(k))*p(loser)))/s_k;
%             grad(winner)=grad(winner)+v;
%             grad(loser)=grad(loser)-v;
%         end

        %% HRA-G gamma
        s_i = s(pair{k}(:,1)); % winner
        s_j = s(pair{k}(:,2)); % loser
        gamma = alpha(k);
        obj = obj - sum(log(exp(s_i * gamma) ./ ...
            (exp(s_i * gamma) + exp(s_j * gamma))))/s_k;
        
        for idx=1:size(pair{k},1)
            i = pair{k}(idx,1); % winner
            j = pair{k}(idx,2); % loser
            v = gamma * exp(gamma * s(j)) ./ ...
                (exp(s(j) * gamma) + exp(s(i) * gamma));
            grad(i) = grad(i) - v;
            grad(j) = grad(j) + v;
        end

        %% HRA-N gamma
%         s_i = s(pair{k}(:,1)); % winner
%         s_j = s(pair{k}(:,2)); % loser
%         gamma = alpha(k);
%         temp_cdf = normcdf((s_i-s_j)*gamma);
%         obj = obj - sum(log(temp_cdf))/s_k;
%         
%         for idx=1:size(pair{k},1)
%             i = pair{k}(idx,1); % winner
%             j = pair{k}(idx,2); % loser
%             v = 1. / temp_cdf(idx) * normpdf((s(i)-s(j))*gamma);
%             grad(i) = grad(i) - v*s(i);
%             grad(j) = grad(j) + v*s(j);
%         end

        %% HRA-E gamma
%         s_i = s(pair{k}(:,1)); % winner
%         s_j = s(pair{k}(:,2)); % loser
%         gamma = alpha(k);
%         x = s_j - s_i;
%         pos = 1/4 * exp(-gamma.*x).*(gamma.*x+2);
%         neg = 1/4 * exp( gamma.*x).*(gamma.*x-2)+1;
%         tot = (sign(x) + 1)/2.*pos + (sign(x) - 1)/(-2).*neg;
%         obj = obj - sum(log(tot))/s_k;
%         
%         for idx=1:size(pair{k},1)
%             i = pair{k}(idx,1); % winner
%             j = pair{k}(idx,2); % loser
%             x = s(j) - s(i); 
%             %-f'(x) * x' = f'(x)(s_i  - s_j)'  f(x) is positive log
%             %likelihood
%             pv = -gamma.*exp(gamma*x+1)./(gamma*x+2);
%             nv = gamma.*exp(gamma*x).*(gamma*x -1)./(exp(gamma.*x).*(gamma*x -2) + 4);
%             tot = (sign(x) + 1)/2.*pv + (sign(x) - 1)/(-2).*nv; 
%             grad(i) = grad(i) + tot;
%             grad(j) = grad(j) - tot;
%         end

%%
    end

    obj=obj+reg_s/2*norm(s,2).^2+reg_alpha/2*norm(alpha,2).^2;
    grad=grad+reg_s.*s;
end