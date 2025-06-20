function [rate_opt, sprad] = hsdm_rate(M, w, param)
    % INPUT
    % M: positive square (N by N) matrix
    % w: positive weights of rates of users
    % param.r_ini : initial vector for algorithm (must be positive)
    % param.K and q : step size mu_k = 1/max(w) if k <= K and (k-K)^(-q)
    % param.stopCri : stopping criterion for algorithm
    % param.maxIter: number of maximum iteration
    %
    % OUTPUT
    % rate_opt: optimized rates of users
    % sprad: spectral radius of diag(exp(rate_opt) - 1) * M

    r = param.r_ini;
    gamma = 0;
    beta = max(w); % Lipschitz constant of gradient (can be any positive number)
    mu = 1 / beta; % step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    K = param.K;
    q = param.q;
    stopCri = param.stopCri;
    maxIter = param.maxIter;

    for kk = 1:maxIter

        r_prev = r;
        gamma_prev = gamma;

        [xi, gamma] = eigs(diag(exp(r) - 1) * M, 1, 'largestreal'); %right eigenvector  
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

        if gamma < 1 + stopCri ...
                && abs(gamma - gamma_prev) < stopCri ...
                && abs(w'*r - w'*r_prev) < stopCri
             break
        end
    end

    rate_opt = r;
    sprad = gamma;
end