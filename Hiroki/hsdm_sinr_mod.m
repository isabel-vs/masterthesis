function [outarg] = hsdm_sinr_mod(param)
    % INPUT
    % M_block: block of L positive square (N by N) matrices (NxNxL)
    % w: positive weights of rates of users
    % param.s_ini : initial vector for algorithm (must be positive)
    % param.K and q : step size mu_k  = 1/max(w) if k <= K and (k-K)^(-q)
    % param.stopCri : stopping criterion for algorithm
    % param.maxIter: number of maximum iteration
    
    % OUTPUT
    % sinr_opt: optimized SINR of users
    % sprad: spectral radius of diag(sinr_opt) * M
    
    M_block = param.M_ary;
    s = param.init;
    w = param.w;
    gamma = 0;
    beta = param.beta; % Lipschitz constant of gradient
    mu = 1 / beta; % step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    K = param.K;
    q = param.q;
%     stopCri = param.stopCri;
    stopCri = 1e-8;
    maxIter = param.maxIter;
    L = size(M_block, 3);
    
    sprad_prog = zeros(maxIter,1);
    sumrate_prog = zeros(maxIter,1);
    rate_prog = zeros(maxIter,L);

    for kk = 1:maxIter

        s_prev = s;
        gamma_prev = gamma;

        gamma_max = 0;
        l_max = NaN;
        for l = 1:L
            [~, gamma_l] = eigs(diag(s) * M_block(:,:,l), 1, 'largestreal');
            if gamma_l > gamma_max
                gamma_max = gamma_l;
                l_max = l;
            end
        end
        gamma = gamma_max;
        M = M_block(:,:,l_max);

        [xi, ~] = eigs(diag(s) * M, 1, 'largestreal'); %right eigenvector
        xi = abs(xi);
        [eta, ~] = eigs(M' * diag(s), 1, 'largestreal'); %left eigenvector
        eta = abs(eta);
%         xi = xi / (xi' * eta);
        if gamma > 1
%             g = gamma * xi .* eta ./ s;
            g = (diag(eta)*M* xi)/(eta' * xi);
            s_hat = s - (gamma - 1) * g / (g' * g);
        else
            s_hat = s;
        end

        s_til = max(0, s_hat); %we can disregard b in the simulations

        if kk > K
            mu = 1 / beta * (kk - K) ^(-q);
        end
        s = s_til + mu * w ./ (1 + s_til);
        
        sprad_prog(kk) = gamma;
        r = log(1+s);
        sumrate_prog(kk) = sum(r);
        rate_prog(kk,:) = r;

        if gamma < 1 + stopCri ...
                && abs(gamma - gamma_prev) < stopCri ...
                && abs(w'*log(1+s) - w'*log(1+s_prev)) < stopCri
             break
        end
    end
    
%     radial projection to feasible region
    s = 1/gamma * s;
%     sinr_opt = s;
%     sprad = gamma;

    gamma_max = 0;
%     l_max = NaN;
    for l = 1:L
        [~, gamma_l] = eigs(diag(s) * M_block(:,:,l), 1, 'largestreal');
        if gamma_l > gamma_max
            gamma_max = gamma_l;
%             l_max = l;
        end
    end
    gamma = gamma_max;
    
    r = log(1+s);
    outarg.sumrate_opt = sum(r);
    outarg.rate_opt = r;
    outarg.sumrate = sumrate_prog;
    outarg.sprad = sprad_prog;
    outarg.gamma = gamma;
    outarg.rate = rate_prog;
end