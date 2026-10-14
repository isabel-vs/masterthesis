clear all

load("additional.mat");
load("toyexample.mat");

%%% Model of interference %%%
% f_n(p) = (C(:,n)' * p + sigma(n)) / b(n) %
% b and C must be positive, and sigma is nonnegative %
% load("figure3.mat", "C", "b", "sig");
% b = b';
% sig = sig';
% n_user = length(b);
% p_max = 200;

n_user = 4;
% b = 1e+4*rand(n_user,1);
% C = 1e+4*rand(n_user,n_user);
% sig = 1e+2 * rand(n_user,1);
% p_max = 200;

%%% F(p) = (f_1(p),...,f_N(p)) = M * p + u %%%
% M = diag(1./b) * C';
% u = diag(1./b) * sig;
F = @(p) M * p + u;

%%% Construct input matrices for proposed algorithm for individual power constraint %%%
L = n_user;
% A = zeros(n_user,L);
% for ll=1:L
%     A(ll,ll) =  1 / p_max;
% end
% M_ary = zeros(n_user, n_user, L);
% for ll = 1:L
%     M_ary(:,:,ll) = M + u * A(:,ll)';
% end

w = ones(n_user, 1); %positive weight (priority) of users

% p_init = [1; 0.01; 1]; %must be positive
% p_init = 0.1 + 0.9 * rand(n_user, 1); %must be positive
% p_init = (p_max/max(p_init)) * p_init;
% s_init = p_init ./ F(p_init);
% r_init = log(1 + s_init);

n_alg = 3;
alg = cell(n_alg,1);
maxIter = 2000;
% 
alg{1}.name = 'hsdm_rate_mod';
alg{1}.param.M_ary = M_ary;
alg{1}.param.w = w;
alg{1}.param.init = r_init;
alg{1}.param.K = 0;  % K > 0 (use constant step-size for first K iterations)
alg{1}.param.q = 1;  % 0 < q <= 1 (decay factor of step-size)
alg{1}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)

alg{2}.name = 'hsdm_sinr_mod';
alg{2}.param.M_ary = M_ary;
alg{2}.param.w = w;
alg{2}.param.init = s_init;
alg{2}.param.K = 0;  % K > 0 (use constant step-size for first K iterations)
alg{2}.param.q = 1;  % 0 < q <= 1 (decay factor of step-size)
alg{2}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)
% 
alg{3}.name = 'bca_mod';
alg{3}.param.b = b;
alg{3}.param.C = C;
alg{3}.param.sig = sig;
alg{3}.param.w = w;
alg{3}.param.F = F;
alg{3}.param.p_max = p_max;
alg{3}.param.init = p_init;
alg{3}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)

result = cell(n_alg,1);

for nn=1:n_alg
    fprintf('run %s\n', alg{nn}.name);
    result{nn} = feval(alg{nn}.name, alg{nn}.param);
%     if ~isfield(result{nn}, 'power_opt')
%         result{nn}.power_opt = conv_power(result{nn}.sinr_opt, F);
%     end
end

% [result_sinr, sprad] = feval(alg{2}.name, alg{2}.param);
% result_bcd = feval(alg{3}.name, alg{3}.param);


% figure; clf; hold on
% labels = strings(n_alg,1);
% for nn=1:n_alg
%     plot(result{nn}.sumrate,'-*');
%     labels(nn) = alg{nn}.name;
% end
% legend(labels,'Location','best','Interpreter','none')
% xlabel('iteration')
% ylabel('sum-rate')
% 
% figure; clf; hold on
% for nn=1:n_alg
%     if isfield(result{nn}, 'sprad')
%         plot(result{nn}.sprad,'-*');
% %     end
% end
% legend(labels,'Location','best','Interpreter','none')
% xlabel('iteration')
% ylabel('spectral radius')


% 
% figure; clf; hold on
% subplot(1,2,1); hold on
% p1 = plot(result{1}.sumrate);
% p1.Color = [0,0,128]/255; p1.LineWidth = 1.5; p1.Marker = 'diamond'; p1.MarkerSize = 6;
% p2 = plot(result{2}.sumrate);
% p2.Color = [100,149,237]/255; p2.LineWidth = 1.5; p2.Marker = 'square'; p2.MarkerSize = 6;
% p3 = plot(result{3}.sumrate);
% p3.Color = [255,69,0]/255; p3.LineWidth = 1.5; p3.Marker = 'x'; p3.MarkerSize = 6;
% legend('Algorithm 1 (rate)', 'Algorithm 2 (SINR)', 'BCA')
% xlabel('iteration')
% ylabel('sum-rate')
% 
% subplot(1,2,2); hold on
% p1 = plot(result{1}.sprad);
% p1.Color = [0,0,128]/255; p1.LineWidth = 1.5; p1.Marker = 'diamond'; p1.MarkerSize = 6;
% p2 = plot(result{2}.sprad);
% p2.Color = [100,149,237]/255; p2.LineWidth = 1.5; p2.Marker = 'square'; p2.MarkerSize = 6;
% legend('Algorithm 1 (rate)', 'Algorithm 2 (SINR)')
% xlabel('iteration')
% ylabel('spectral radius')



