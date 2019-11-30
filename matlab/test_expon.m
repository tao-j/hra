clear

x = linspace(-4, 4);
y = expon(x, 1);
plot(x, y);

% hold on
% yd = expon_d(x, 1);
% plot(x, yd);

function tot=expon(x, gamma)
    pos = 1/4 * exp(-gamma.*x).*(gamma.*x+2);
    neg = 1/4 * exp( gamma.*x).*(gamma.*x-2)+1;
    tot = (sign(x) + 1)/2.*pos + (sign(x) - 1)/(-2).*neg;             
end

function grad=expon_d(x, gamma)
    pv = -gamma.*exp(gamma*x+1)./(gamma*x+2);
    nv = gamma.*exp(gamma*x).*(gamma*x -1)./(exp(gamma.*x).*(gamma*x -2) + 4);
    tot = (sign(x) + 1)/2.*pv + (sign(x) - 1)/(-2).*nv; 
    grad = tot;
end

%%