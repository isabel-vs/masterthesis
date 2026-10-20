function r_out = P_rad(r, M)
    % Range: [0,1]
    a_new = 0.5;
    
    % Calculate initial spectral radius
    try
        gamma = eigs(diag(exp(a_new * r) - 1) * M, 1, 'largestreal');
    catch
        gamma = max(real(eig(diag(exp(a_new * r) - 1) * M)));
    end
    
    thresh = 1e-8;
    L_bound = 0;
    R_bound = 1;
    
    % Bisection Loop
    while (gamma < 1 - thresh) || (gamma > 1 + thresh)
        a_old = a_new;
        
        if gamma > 1
            a_new = a_old - 0.5 * (a_old - L_bound);
            R_bound = a_old;
        else % gamma < 1
            a_new = a_old + 0.5 * (R_bound - a_old);
            L_bound = a_old;
        end
        
        try
            gamma = eigs(diag(exp(a_new * r) - 1) * M, 1, 'largestreal');
        catch
            gamma = max(real(eig(diag(exp(a_new * r) - 1) * M)));
        end
    end
    
    r_out = a_new * r;
end