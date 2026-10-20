function p_new = power_fixpoint(n_user, F, p_max, s)
    % Initialize randomly between 0 and p_max
    % MATLAB's rand produces values in (0, 1)
    p_new = p_max * rand(n_user, 1);
    
    thresh = 1e-8;
    max_iter = 100000;
    
    for iter = 1:max_iter
        p_old = p_new;
        
        % In MATLAB, np.diag(s) @ F(...) is equivalent to element-wise s .* F(...)
        % p_old(:) ensures the input is a column vector (similar to flatten)
        p_new = s .* F(p_old(:));

        % Check for convergence
        % np.abs(np.max(...)) in Python usually refers to the infinity norm
        if max(abs(p_new - p_old)) < thresh
            break;
        end
    end
end