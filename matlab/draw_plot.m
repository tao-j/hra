close all;

figure('position', [200 650 400 300]);
hold all;
cutoff=17;
box on;
ylim([0. 1.9])
plot(1:cutoff, df(1:cutoff, 1), '-o', 'LineWidth',1, 'color', 'r');
plot(1:cutoff, df(1:cutoff, 2), '-s', 'LineWidth',1, 'color', 'b');
plot(1:cutoff, df(1:cutoff, 3), '-x', 'LineWidth',1, 'color', 'm');
% ylabel('$||$\boldmath$s_{t}$- \boldmath$s^*$ $||_2$','Interpreter','latex')
ylabel('$\|\mathbf{s}_{t}-\mathbf{s}^*\|_2$','FontName','Times','FontSize',12,'interpreter','latex');
xlabel('Number of iteration $t$','Interpreter','latex')
legend('$\gamma_A=5, \gamma_B=0.25$','$\gamma_A=5, \gamma_B=1$', '$\gamma_A=5, \gamma_B=2.5$','Interpreter','latex')
saveas(gcf,['./figure/HBTL','_s','.pdf']);


figure('position', [200 1050 400 300]);
hold all;
cutoff=17;
box on;
ylim([1. 7.8])
plot(1:cutoff, df(1:cutoff, 4), '-o', 'LineWidth',1, 'color', 'r');
plot(1:cutoff, df(1:cutoff, 5), '-s', 'LineWidth',1, 'color', 'b');
plot(1:cutoff, df(1:cutoff, 6), '-x', 'LineWidth',1, 'color', 'm');
ylabel('$\|$\boldmath$\gamma_{t}$-$\boldmath{\gamma}^*\|_2$','FontName','Times','FontSize',12,'interpreter','latex');% ylabel('||\bf{\gamma}-\bf{\gamma}||_2','FontName','Times','FontSize',12,'interpreter','tex');
xlabel('Number of iteration $t$','Interpreter','latex') 
legend('$\gamma_A=5, \gamma_B=0.25$','$\gamma_A=5, \gamma_B=1$', '$\gamma_A=5, \gamma_B=2.5$','Interpreter','latex')
saveas(gcf,['./figure/HBTL','_gamma','.pdf']);


figure('position', [600 650 400 300]);
hold all;
cutoff=22;
box on;
ylim([0. 1.9])
plot(1:cutoff, df(1:cutoff, 7), '-o', 'LineWidth',1, 'color', 'r');
plot(1:cutoff, df(1:cutoff, 8), '-s', 'LineWidth',1, 'color', 'b');
plot(1:cutoff, df(1:cutoff, 9), '-x', 'LineWidth',1, 'color', 'm');
ylabel('$\|\mathbf{s}_{t}-\mathbf{s}^*\|_2$','FontName','Times','FontSize',12,'interpreter','latex');
xlabel('Number of iteration $t$','Interpreter','latex')
legend('$\gamma_A=5, \gamma_B=0.25$','$\gamma_A=5, \gamma_B=1$', '$\gamma_A=5, \gamma_B=2.5$','Interpreter','latex')
saveas(gcf,['./figure/HTCV','_s','.pdf']);

figure('position', [600 1050 400 300]);
hold all;
cutoff=22;
box on;
ylim([1. 7.8])
plot(1:cutoff, df(1:cutoff, 10), '-o', 'LineWidth',1, 'color', 'r');
plot(1:cutoff, df(1:cutoff, 11), '-s', 'LineWidth',1, 'color', 'b');
plot(1:cutoff, df(1:cutoff, 12), '-x', 'LineWidth',1, 'color', 'm');
ylabel('$\|\boldmath{\gamma}_{t}-\boldmath{\gamma}^*\|_2$','FontName','Times','FontSize',12,'interpreter','latex');
% ylabel('||\bf{\gamma}-\bf{\gamma}||_2','FontName','Times','FontSize',12,'interpreter','tex');
xlabel('Number of iteration $t$','Interpreter','latex') 
legend('$\gamma_A=5, \gamma_B=0.25$','$\gamma_A=5, \gamma_B=1$', '$\gamma_A=5, \gamma_B=2.5$','Interpreter','latex')
saveas(gcf,['./figure/HTCV','_gamma','.pdf']);