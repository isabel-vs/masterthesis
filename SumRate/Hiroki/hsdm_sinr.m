function [sinr_opt, sprad] = hsdm_sinr(M, w, param)
    % INPUT
    % M: positive square (N by N) matrix
    % w: positive weights of rates of users
    % param.r_ini : initial vector for algorithm (must be positive)
    % param.K and q : tep size mu_k  = 1/max(w) if k <= K and (k-K)^(-q)
    % param.stopCri : stopping criterion for algorithm
    % param.maxIter: number of maximum iteration
    %
    % OUTPUT
    % sinr_opt: optimized SINR of users
    % sprad: spectral radius of diag(sinr_opt) * M

    s = param.s_ini;
    gamma = 0;
    beta = max(w); % Lipschitz constant of gradient
    mu = 1 / beta; % step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    K = param.K;
    q = param.q;
    stopCri = param.stopCri;
    maxIter = param.maxIter;

    for kk = 1:maxIter

        s_prev = s;
        gamma_prev = gamma;

        [xi, gamma] = eigs(diag(s) * M, 1, 'largestreal'); %right eigenvector
        xi = abs(xi);
        [eta, ~] = eigs(M' * diag(s), 1, 'largestreal'); %left eigenvector
        eta = abs(eta);
        xi = xi / (xi' * eta);
        if gamma > 1
            g = gamma * xi .* eta ./ s;
            s_hat = s - (gamma - 1) * g / (g' * g);
        else
            s_hat = s;
        end

        s_til = max(0, s_hat); %we can disregard b in the simulations

        if kk > K
            mu = 1 / beta * (kk - K) ^(-q);
        end
        s = s_til + mu * w ./ (1 + s_til);

        if gamma < 1 + stopCri ...
                && abs(gamma - gamma_prev) < stopCri ...
                && abs(w'*log(1+s) - w'*log(1+s_prev)) < stopCri
             break
        end
    end

    sinr_opt = s;
    sprad = gamma;
end