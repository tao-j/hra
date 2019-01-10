function [s_new,alpha_new, obj, iter]=alter(s, alpha, pair, para)
    
    
    maxiter=getOpt(para, 'maxiter', 100);
    verbose=getOpt(para, 'verbose', true);
    tol=getOpt(para, 'tol', 1e-2);
    opt_method=getOpt(para, 'opt_method', 's->a+GD');
    lr=getOpt(para, 'lr', 10e-3);
        
    obj=zeros(maxiter, 1);
    opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 500, 'optTol', 1e-5, 'progTol', 1e-7);
    opt_a=struct('method', 'newton', 'verbose', 0);
    interval=1;
    
    for iter=1 : maxiter
        
        switch opt_method
            case 'a->s+lbfgs'
                [alpha_new, obj(iter)]=minFunc(@func_alpha, alpha, opt_s, s, para, pair);
                [s_new, obj_s]=minFunc(@func_s, s, opt_s,  alpha_new, para, pair);

            case 'a->s+newton'
                p=@(x)func_alpha(x, s, para, pair);
                [alpha_new, obj(iter)]=minConf_TMP(p, alpha, -50, 50, opt_a);
                [s_new, obj_s]=minFunc(@func_s, s, opt_s,  alpha_new, para, pair);

            case 's->a+lbfgs'
                [s_new, obj_s]=minFunc(@func_s, s, opt_s,  alpha, para, pair);
                [alpha_new, obj(iter)]=minFunc(@func_alpha, alpha, opt_s, s, para, pair);

            case 's->a+newton'
                [s_new, obj_s]=minFunc(@func_s, s, opt_s,  alpha, para, pair);
                p=@(x)func_alpha(x, s_new, para, pair);
                [alpha_new, obj(iter)]=minConf_TMP(p, alpha, -50, 50, opt_a);

            case 's->a+newton+crowdbt'
                [s_new, obj_s]=minFunc(@func_s, s, opt_s,  alpha, para, pair);
                p=@(x)func_alpha(x, s_new, para, pair);
                [alpha_new, obj(iter)]=minConf_TMP(p, alpha, 0, 1, opt_a);

            case 'a->s+GD'
                interval=30;
                [obj(iter), grad_a, ~] = func_alpha(alpha, s, para, pair);
                alpha_new = alpha - lr*grad_a;

                [obj_s, grad_s] = func_s(s, alpha_new, para, pair);
                s_new = s - lr*grad_s;
                
            case 's->a+GD'
                interval=30;
                [obj_s, grad_s] = func_s(s, alpha, para, pair);
                s_new = s - lr*grad_s;
                
                [obj(iter), grad_a, ~] = func_alpha(alpha, s_new, para, pair);
                alpha_new = alpha - lr*grad_a;
        end
        %%
%         if (verbose && mod(iter, interval) == 0)
%             fprintf('Iter %d, obj: %.4f, obj_s: %.3f, mean_alpha: %.2f\n', iter,  obj(iter), obj_s, mean(alpha_new));
%         end
        
%         if ( iter > 10 && abs((obj(iter) - obj(iter-1))/obj(iter)) <tol)
        if (iter == 300)
                lr = lr / 5;
                fprintf('div lr by 5\n');
        else
            alpha=alpha_new;
            s=s_new;
        end
    end     
    
    obj=obj(1:iter); 
end


