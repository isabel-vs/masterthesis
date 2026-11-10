clear all

%%% Model of interference %%%
% f_n(p) = (C(:,n)' * p + sigma(n)) / b(n) %
% b and C must be positive, and sigma is nonnegative %

n_user = 2;
p_max = 1;
maxIter = 5000;

g1 = 2;
g2 = 1;
noise = 0.01;

n_points = 5000;

p1 = p_max * rand(n_points,1);
p2 = p_max * rand(n_points,1);

s1 = p1*g1 ./ (p1+p2*g2 + noise);
s2 = p2*g2 ./ (p2+p1*g1 + noise);
r1 = log(1 + (p1*g1 ./ (p1+p2*g2 + noise)));
r2 = log(1 + (p2*g2 ./ (p2+p1*g1 + noise)));

%%% F(p) = (f_1(p),...,f_N(p)) = M * p + u %%%
M = [[1/g1, g2/g1]
    [g1/g2, 1/g2]];
u = [noise/g1, noise/g2]';

F = @(p) M * p + u;

%%% Construct input matrices for proposed algorithm for individual power constraint %%%
L = n_user;
A = zeros(n_user,L);
for ll=1:L
    A(ll,ll) =  1 / p_max;
end
M_ary = zeros(n_user, n_user, L);
for ll = 1:L
    M_ary(:,:,ll) = M + u * A(:,ll)';
end

w = ones(n_user, 1); %positive weight (priority) of users
p_bca = ones(n_user,1).*p_max;
s_init = ones(n_user, 1)*2;
r_init = ones(n_user, 1)*2;

n_alg = 4;
alg = cell(n_alg,1);
% 
alg{1}.name = 'hsdm_rate_mod';
alg{1}.param.M_ary = M_ary;
alg{1}.param.w = w;
alg{1}.param.beta = 1;
alg{1}.param.init = r_init;
alg{1}.param.K = 0;  % K > 0 (use constant step-size for first K iterations)
alg{1}.param.q = 0.999;  % 0 < q <= 1 (decay factor of step-size)
alg{1}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)

alg{2}.name = 'hsdm_sinr_mod';
alg{2}.param.M_ary = M_ary;
alg{2}.param.w = w;
alg{2}.param.beta = 1;
alg{2}.param.init = s_init;
alg{2}.param.K = 0;  % K > 0 (use constant step-size for first K iterations)
alg{2}.param.q = 0.999;  % 0 < q <= 1 (decay factor of step-size)
alg{2}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)

alg{3}.name = 'hsdm_cut';
alg{3}.param.M_ary = M_ary;
alg{3}.param.w = w;
alg{3}.param.beta = 1;
alg{3}.param.init = r_init;
alg{3}.param.K = 0;  % K > 0 (use constant step-size for first K iterations)
alg{3}.param.q = 0.999;  % 0 < q <= 1 (decay factor of step-size)
alg{3}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)

alg{4}.name = 'hsdm_cut_s';
alg{4}.param.M_ary = M_ary;
alg{4}.param.w = w;
alg{4}.param.beta = 1;
alg{4}.param.init = r_init;
alg{4}.param.K = 0;  % K > 0 (use constant step-size for first K iterations)
alg{4}.param.q = 0.999;  % 0 < q <= 1 (decay factor of step-size)
alg{4}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)


result = cell(n_alg,1);
r_opt = zeros(n_alg,n_user);
s_opt = zeros(n_alg,n_user);

exec_time = zeros(n_alg, 1);

for nn=1:n_alg
    fprintf('run %s\n', alg{nn}.name);
    tic;
    result{nn} = feval(alg{nn}.name, alg{nn}.param);
    exec_time(nn) = toc;
%     if ~isfield(result{nn}, 'power_opt')
%         result{nn}.power_opt = conv_power(result{nn}.sinr_opt, F);
%     end
    fprintf('sumrate %f\n', result{nn}.sumrate_opt);
    if isfield(result{nn}, 'sprad')
        fprintf('sprad (before Prad) %f\n', result{nn}.sprad(end));
    end
    if isfield(result{nn}, 'gamma')
        fprintf('sprad (after Prad) %f\n', result{nn}.gamma);
    end
    r_opt(nn,:) = result{nn}.rate_opt;
    s_opt(nn,:) = exp(r_opt(nn,:))-1;
end

% plot results
%sinr region
figure('Position', [100, 100, 700, 500]);
hold on

scatter(s1, s2, 2, 'filled', ...
    'MarkerFaceAlpha', 0.25, ...
    'MarkerEdgeAlpha', 0.25);

colors = lines(size(s_opt, 1));
markers = {'o', 's', '^', 'd'};

for i = 1:size(s_opt, 1)
    scatter(s_opt(i,1), s_opt(i,2), 60, colors(i,:), 'filled', ...
        markers{i}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf(alg{i}.name));
end


xlabel('$s_1$', 'Interpreter', 'latex');
ylabel('$s_2$', 'Interpreter', 'latex');
title(sprintf('Point cloud: g1=%g, g2=%g', g1, g2));

grid on;
ax = gca;
ax.GridAlpha = 0.2;

set(gca, 'LooseInset', get(gca, 'TightInset'));

%rate region
figure('Position', [100, 100, 700, 500]);
hold on

scatter(r1, r2, 2, 'filled', ...
    'MarkerFaceAlpha', 0.25, ...
    'MarkerEdgeAlpha', 0.25);

for i = 1:size(r_opt, 1)
    scatter(r_opt(i,1), r_opt(i,2), 60, colors(i,:), 'filled', ...
        markers{i}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf(alg{i}.name));
end

xlabel('$r_1$', 'Interpreter', 'latex');
ylabel('$r_2$', 'Interpreter', 'latex');
title(sprintf('Point cloud: g1=%g, g2=%g', g1, g2));

grid on;
ax = gca;
ax.GridAlpha = 0.2;
legend('Location', 'best');

set(gca, 'LooseInset', get(gca, 'TightInset'));

fprintf('\nExecution time comparison:\n');
for nn = 1:n_alg
    fprintf('%-20s %10.4f seconds\n', alg{nn}.name, exec_time(nn));
end





