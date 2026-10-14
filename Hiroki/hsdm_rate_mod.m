function [outarg] = hsdm_rate_mod(param)
    % INPUT
    % M_block: block of L positive square (N by N) matrices (NxNxL)
    % w: positive weights of rates of users
    % param.r_ini : initial vector for algorithm (must be positive)
    % param.K and q : step size mu_k = 1/max(w) if k <= K and (k-K)^(-q)
    % param.stopCri : stopping criterion for algorithm
    % param.maxIter: number of maximum iteration
    %
    % OUTPUT
    % rate_opt: optimized rates of users
    % sprad: spectral radius of diag(exp(rate_opt) - 1) * M
    
    M_block = param.M_ary;
    r = param.init;
    w = param.w;
    gamma = 0;
    beta = max(w); % Lipschitz constant of gradient (can be any positive number)
    mu = 1 / beta; % step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    K = param.K;
    q = param.q;
%     stopCri = param.stopCri;
    stopCri = 1e-8;
    maxIter = param.maxIter;
    L = size(M_block, 3);
    
    sprad_prog = zeros(maxIter,1);
    sumrate_prog = zeros(maxIter,1);
    
%     rate_track = [];

    for kk = 1:maxIter
        
        r_prev = r;
        gamma_prev = gamma;

        gamma_max = 0;
        l_max = NaN;
        for l = 1:L
            [~, gamma_l] = eigs(diag(exp(r) - 1) * M_block(:,:,l), 1, 'largestreal');
            if gamma_l > gamma_max
                gamma_max = gamma_l;
                l_max = l;
            end
        end
        gamma = gamma_max;
        M = M_block(:,:,l_max);

        [xi, ~] = eigs(diag(exp(r) - 1) * M, 1, 'largestreal'); %right eigenvector  
        xi = abs(xi);
        [eta, ~] = eigs(M' * diag(exp(r) - 1), 1, 'largestreal'); %left eigenvector
        eta = abs(eta);
        xi = xi / (xi' * eta);
        if gamma > 1
            g = gamma * xi .* eta .* exp(r) ./ (exp(r) - 1);
            r_hat = r - (gamma - 1) * g / (g' * g);
        else
            r_hat = r;
        end
        r_til = max(0, r_hat); %we can disregard b in the simulations

        if kk > K
            mu = 1 / beta * (kk - K) ^(-q);
        end
        r = r_til + mu * w;
        
%         rate_track = [rate_track r];
        
        sprad_prog(kk) = gamma;
        sumrate_prog(kk) = sum(r);

        if gamma < 1 + stopCri ...
                && abs(gamma - gamma_prev) < stopCri ...
                && abs(w'*r - w'*r_prev) < stopCri
             break
        end
    end
    
%     rate_opt = r;
%     sprad = gamma;
%     sumrate_opt = sum(r);
    outarg.sumrate_opt = sum(r);
    outarg.sumrate = sumrate_prog;
    outarg.sprad = sprad_prog;

end