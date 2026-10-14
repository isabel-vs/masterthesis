function [outarg] = bca_mod(param)
    
    sig = param.sig;
    b = param.b;
    C = param.C;
    F = param.F;
    p_max = param.p_max;

    p = param.init;
    maxIter = param.maxIter;
    outarg.sumrate = zeros(maxIter,1);
    N = length(b);
    u = zeros(N,1);
    d = zeros(N,1);

    for kk = 1: maxIter

        for nn = 1:N
            tmp1 = sqrt(b(nn) * p(nn));
            tmp2 = b(nn) * p(nn) + C(:,nn)' * p + sig(nn);
            u(nn) = tmp1 / tmp2;
            d(nn) = 1 / (u(nn)^2 * tmp2 - 2 * u(nn) * tmp1 + 1);
        end

        for nn = 1:N
            tmp1 = b(nn) * d(nn)^2 * u(nn)^2;
%             tmp2 = (b(nn) * d(nn) * u(nn)^2 + (d .* (u .^2))' * C(:,nn))^2;
            tmp2 = (b(nn) * d(nn) * u(nn)^2 + (d .* (u.^2))' * C(nn,:)')^2;
            p(nn) = min(tmp1 / tmp2, p_max);
        end
        outarg.sumrate(kk) = ones(N,1)' * log(1 + p ./ F(p));
    end

    outarg.power_opt = p;
    outarg.sumrate_opt = ones(N,1)' * log(1 + p ./ F(p));
end