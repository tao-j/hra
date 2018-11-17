function [s_new,alpha_new, obj, iter]=alter(s, alpha, pair, para)
    
    
    maxiter=getOpt(para, 'maxiter', 100);
    verbose=getOpt(para, 'verbose', true);
    tol=getOpt(para, 'tol', 1e-2);
        
    obj=zeros(maxiter, 1);
    opt_s=struct('Method', 'lbfgs', 'DISPLAY', 0, 'MaxIter', 500, 'optTol', 1e-5, 'progTol', 1e-7);
    opt_a=struct('method', 'newton', 'verbose', 0);
    
    for iter =1 : maxiter
        
        % optimization w.r.t. alpha
        p=@(x)func_alpha(x, s, para, pair);
        [alpha_new, obj(iter)]=minConf_TMP(p, alpha, -50, 50, opt_a);
%         [alpha_new, obj(iter)]=minFunc(@func_alpha, alpha, opt_s, s, para, pair);
        
        % optimization w.r.t. s
        [s_new, obj_s]=minFunc(@func_s, s, opt_s,  alpha_new, para, pair);
        
%         [alpha_new, obj(iter)]=minFunc(@func_alpha, alpha, opt_s, s_new, para, pair);
        
        
        if (verbose)
            fprintf('Iter %d, obj: %.4f, obj_s: %.3f, mean_alpha: %.2f\n', iter,  obj(iter), obj_s, mean(alpha_new));
        end
        
        if ( iter > 10 && obj(iter) - obj(iter-1) <tol)
            break;
        else
            alpha=alpha_new;
            s=s_new;
        end
    end     
    
    obj=obj(1:iter);      
end


