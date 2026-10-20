clear all

% load("additional.mat");
% load("toyexample.mat");
% load("demo_nicecut.mat");
load("demo_invZ.mat");
% load("my_data.mat");

%%% Model of interference %%%
% f_n(p) = (C(:,n)' * p + sigma(n)) / b(n) %
% b and C must be positive, and sigma is nonnegative %
% load("figure3.mat", "C", "b", "sig");
b = b';
sig = sig;
n_user = length(b);
p_max = 200;
maxIter = 50;

% n_user = 8;
% b = 1e+4*rand(n_user,1);
% C = 1e+4*rand(n_user,n_user);
% sig = 1e+2 * rand(n_user,1);
% p_max = 200;

% b = [3.46235566e+10, 1.28888203e+10, 7.97209004e+09, 2.91349628e+09]';
% C = [[2.36819729e+10, 8.51545769e+07, 1.97993559e+08, 1.76980690e+07]
%        [8.51607266e+07, 1.03939006e+09, 3.69061086e+07, 3.15646177e+07]
%        [1.97999905e+08, 3.69054326e+07, 5.24552092e+08, 2.48283037e+07]
%        [1.77052682e+07, 3.15553092e+07, 2.48240245e+07, 1.77499356e+08]];
% sig = [186073.56313103, 113528.73312086,  89286.40866014,  53977.11515151]';

% b = [8.26411741e+06, 2.08060501e+05, 6.12335197e+04, 1.13129525e+08, 1.81259435e+06, 2.68894405e+06, 1.00104409e+07, 1.01413566e+07]';
% C = [[5.43966311e+06, 5.02312192e+05, 1.66251665e+05, 6.59413883e+05,  5.80307178e+05, 7.37941648e+05, 1.83476815e+06, 6.79413475e+05]
%        [5.01934953e+05, 8.16052882e+04, 2.72434686e+04, 1.35194267e+05,7.97143832e+04, 1.49747686e+05, 2.46618653e+05, 9.09207036e+04]
%        [1.66355869e+05, 2.72956834e+04, 1.64543023e+04, 4.70787070e+05,  5.51531393e+04, 8.22601715e+04, 1.30766271e+05, 8.47272101e+04]
%        [6.59244299e+05, 1.35058196e+05, 4.70566281e+05, 1.13542196e+08,  7.45139810e+06, 2.21691763e+06, 9.86073329e+06, 2.79295852e+06]
%        [5.80456693e+05, 7.96804461e+04, 5.51994664e+04, 7.45172007e+06, 8.55917885e+05, 2.41718130e+05, 9.58885591e+05, 5.71050853e+04]
%        [7.37887065e+05, 1.49669631e+05, 8.22439515e+04, 2.21705121e+06, 2.41459242e+05, 1.12983554e+06, 1.14945082e+06, 9.12522706e+05]
%        [1.83531951e+06, 2.46773294e+05, 1.30568568e+05, 9.86120511e+06,  9.58969612e+05, 1.14961845e+06, 3.93537656e+06, 1.45772448e+06]
%        [6.79427071e+05, 9.08876869e+04, 8.47259354e+04, 2.79165650e+06,  5.71585102e+04, 9.12685054e+05, 1.45713315e+06, 3.37429686e+06]];
% sig = [ 2874.81664987,   456.16777916,   247.42285211, 10636.25384112,  1346.21262461,  1639.81097799,  3163.77638439,  3184.66017993]';

%%% F(p) = (f_1(p),...,f_N(p)) = M * p + u %%%
M = diag(1./b) * C';
u = diag(1./b) * sig;
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
% 
% p_init = [1; 0.01; 1]; %must be positive
p_bca = ones(n_user,1).*p_max;
p_bca(3) = 0;
p_bca(4) = 0;
% p_init = 0.1 + 0.9 * rand(n_user, 1); %must be positive
p_init = (p_max/max(p_init)) * p_init;
% s_init = p_init ./ F(p_init);
% r_init = log(1 + s_init);
s_init = ones(n_user, 1)*10;
r_init = ones(n_user, 1)*5;
% s_init = exp(r_init)-1;

n_alg = 5;
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
alg{2}.param.beta = 0.9;
alg{2}.param.init = s_init;
alg{2}.param.K = 0;  % K > 0 (use constant step-size for first K iterations)
alg{2}.param.q = 0.8;  % 0 < q <= 1 (decay factor of step-size)
alg{2}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)

alg{3}.name = 'hsdm_cut';
alg{3}.param.M_ary = M_ary;
alg{3}.param.w = w;
alg{3}.param.beta = 1;
alg{3}.param.init = r_init;
alg{3}.param.K = 0;  % K > 0 (use constant step-size for first K iterations)
alg{3}.param.q = 0.999;  % 0 < q <= 1 (decay factor of step-size)
alg{3}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)
% 

alg{4}.name = 'bca_mod';
alg{4}.param.b = b;
alg{4}.param.C = C;
alg{4}.param.sig = sig;
alg{4}.param.w = w;
alg{4}.param.F = F;
alg{4}.param.p_max = p_max;
alg{4}.param.init = p_init;
alg{4}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)

alg{5}.name = 'bca_mod';
alg{5}.param.b = b;
alg{5}.param.C = C;
alg{5}.param.sig = sig;
alg{5}.param.w = w;
alg{5}.param.F = F;
alg{5}.param.p_max = p_max;
alg{5}.param.init = p_bca;
alg{5}.param.maxIter = maxIter; %number of max iteration (need to be sufficiently large)

result = cell(n_alg,1);

for nn=1:n_alg
    fprintf('run %s\n', alg{nn}.name);
    result{nn} = feval(alg{nn}.name, alg{nn}.param);
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
% subplot(1,3,1); hold on
% p1 = plot(result{1}.sumrate);
% p1.Color = [0,0,128]/255; p1.LineWidth = 1; 
% % p1.Marker = 'diamond'; p1.MarkerSize = 6;
% line1 = yline(result{1}.sumrate_opt, ':', 'LineWidth', 1.2, 'Color', p1.Color);
% line1.HandleVisibility = 'off'; % Hide from legend
% p2 = plot(result{2}.sumrate);
% p2.Color = [100,149,237]/255; p2.LineWidth = 1; 
% % p2.Marker = 'square'; p2.MarkerSize = 6;
% line2 = yline(result{2}.sumrate_opt, ':', 'LineWidth', 1.2, 'Color', p2.Color);
% line2.HandleVisibility = 'off';
% p3 = plot(result{3}.sumrate);
% p3.Color = [255,69,0]/255; p3.LineWidth = 1; 
% % p3.Marker = 'x'; p3.MarkerSize = 6;
% line3 = yline(result{3}.sumrate_opt, ':', 'LineWidth', 1.2, 'Color', p3.Color);
% line3.HandleVisibility = 'off';
% p4 = plot(result{4}.sumrate);
% p4.Color = [0,0,0]/255; p4.LineWidth = 1; 
% % p4.Marker = 'x'; p4.MarkerSize = 6;
% p5 = plot(result{5}.sumrate);
% p5.Color = [0,0,255]/255; p5.LineWidth = 1; 
% % p5.Marker = 'x'; p5.MarkerSize = 6;
% legend('Algorithm 1 (rate)', 'Algorithm 2 (SINR)', 'Algorithm 3 (cut)', 'BCD(a)', 'BCD(b)')
% xlabel('iteration')
% ylabel('sum-rate')
% 
% subplot(1,3,2); hold on
% % p1 = plot((result{1}.sumrate- result{1}.sumrate_opt));
% p1 = plot(result{1}.sprad);
% p1.Color = [0,0,128]/255; p1.LineWidth = 1; 
% % p1.Marker = 'diamond'; p1.MarkerSize = 6;
% p2 = plot(result{2}.sprad);
% p2.Color = [100,149,237]/255; p2.LineWidth = 1; 
% % p2.Marker = 'square'; p2.MarkerSize = 6;
% % p3 = plot((result{3}.sumrate- result{3}.sumrate_opt));
% p3 = plot(result{3}.sprad);
% p3.Color = [255,69,0]/255; p3.LineWidth = 1; 
% % p3.Marker = 'square'; p3.MarkerSize = 6;
% legend('Algorithm 1 (rate)', 'Algorithm 2 (SINR)', 'Algorithm 3 (cut)')
% xlabel('iteration')
% ylabel('spectral radius')
% ylim([0, inf])
% 
% r_R = result{1}.rate_opt;
% s_R = exp(r_R)-1;
% r_S = result{2}.rate_opt;
% s_S = exp(r_S)-1;
% r_C = result{3}.rate_opt;
% s_C = exp(r_C)-1;
% 
% subplot(1,3,3); hold on
% p1 = plot(vecnorm(result{1}.rate - r_R', 2, 2));
% % p1 = plot(result{1}.sprad);
% p1.Color = [0,0,128]/255; p1.LineWidth = 1; 
% % p1.Marker = 'diamond'; p1.MarkerSize = 6;
% % p2 = plot(result{2}.sprad);
% p2 = plot(vecnorm(result{2}.rate - r_S', 2, 2));
% p2.Color = [100,149,237]/255; p2.LineWidth = 1; 
% % p2.Marker = 'square'; p2.MarkerSize = 6;
% p3 = plot(vecnorm(result{3}.rate - r_C', 2, 2));
% % p3 = plot(result{3}.sprad);
% p3.Color = [255,69,0]/255; p3.LineWidth = 1; 
% % p3.Marker = 'square'; p3.MarkerSize = 6;
% legend('Algorithm 1 (rate)', 'Algorithm 2 (SINR)', 'Algorithm 3 (cut)')
% xlabel('iteration')
% ylabel('distance to optimum')
% ylim([0, inf])
% % , 'Algorithm 2 (SINR)', 

% 1. Settings for "Bigger" look
f_size = 10;      % Font size for labels, legends, and ticks
line_w = 1.5;       % Thickness of the plot lines
fig_width = 30;   % Width in cm (broad)
fig_height = 10;  % Height in cm (short)

fig = figure(1); 
clf;
fig.Units = 'centimeters';
fig.Position = [1, 1, fig_width, fig_height]; 

% 2. Layout (using tiledlayout for 2019b)
t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Tile 1 (Sum-rate) ---
nexttile; hold on
p1 = plot(result{1}.sumrate, 'Color', [0,0,128]/255, 'LineWidth', line_w);
line1 = yline(result{1}.sumrate_opt, ':', 'LineWidth', 1.5, 'Color', p1.Color);
line1.HandleVisibility = 'off';
% p2 = plot(result{2}.sumrate, 'Color', [100,149,237]/255, 'LineWidth', line_w);
% line2 = yline(result{2}.sumrate_opt, ':', 'LineWidth', 1.5, 'Color', p2.Color);
% line2.HandleVisibility = 'off';
p3 = plot(result{3}.sumrate, 'Color', [255,69,0]/255, 'LineWidth', line_w);
line3 = yline(result{3}.sumrate_opt, ':', 'LineWidth', 1.5, 'Color', p3.Color);
line3.HandleVisibility = 'off';
p4 = plot(result{4}.sumrate, 'Color', [0,0,0]/255, 'LineWidth', line_w);
p5 = plot(result{5}.sumrate, 'Color', [0,0,255]/255, 'LineWidth', line_w);

xlabel('Iteration', 'FontSize', f_size); 
ylabel('Sum-rate [nats/s/Hz]', 'FontSize', f_size);
legend('Alg 1 (rate)', 'Alg 2 (cut)', 'BCD(a)', 'BCD(b)', 'FontSize', f_size);
grid on;

% --- Tile 2 (Spectral Radius) ---
nexttile; hold on
plot(result{1}.sprad, 'Color', [0,0,128]/255, 'LineWidth', line_w);
% plot(result{2}.sprad, 'Color', [100,149,237]/255, 'LineWidth', line_w);
plot(result{3}.sprad, 'Color', [255,69,0]/255, 'LineWidth', line_w);

xlabel('Iteration', 'FontSize', f_size); 
ylabel('Spectral radius', 'FontSize', f_size);
legend('Alg 1 (rate)', 'Alg 2 (SINR)', 'Alg 3 (cut)', 'FontSize', f_size);
grid on;
ylim([0, inf]);

% % --- Tile 3 (Distance) ---
nexttile; hold on
r_R = result{1}.rate_opt; r_S = result{2}.rate_opt; r_C = result{3}.rate_opt;
plot(vecnorm(result{1}.rate - r_R', 2, 2), 'Color', [0,0,128]/255, 'LineWidth', line_w);
% plot(vecnorm(result{2}.rate - r_S', 2, 2), 'Color', [100,149,237]/255, 'LineWidth', line_w);
plot(vecnorm(result{3}.rate - r_C', 2, 2), 'Color', [255,69,0]/255, 'LineWidth', line_w);

xlabel('Iteration', 'FontSize', f_size); 
ylabel('Distance to optimum', 'FontSize', f_size);
legend('Alg 1 (rate)', 'Alg 3 (cut)', 'FontSize', f_size);
grid on;
ylim([0, inf]);

% 3. Apply global Font Size to all Axes (the tick numbers)
set(findall(fig,'type','axes'), 'FontSize', f_size);

% 4. PDF Export Prep
fig.PaperUnits = 'centimeters';
fig.PaperPositionMode = 'manual';
fig.PaperSize = [fig.Position(3), fig.Position(4)];
fig.PaperPosition = [0, 0, fig.Position(3), fig.Position(4)];

% 5. Save as PDF
print(fig, 'Results_Consistent_Large.pdf', '-dpdf', '-painters');

r_R = result{1}.rate_opt;
sr_R = result{1}.sumrate_opt
s_R = exp(r_R)-1;
p_R = power_fixpoint(n_user, F, p_max, s_R);
p_Rn = p_max/max(p_R) * p_R
s_Rn = p_Rn ./ F(p_Rn);
r_Rn = log(s_Rn +1);



