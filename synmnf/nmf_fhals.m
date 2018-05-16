function [W, Htt] = nmf_fhals(A, k)

[m, n] = size(A);

W = max(0.0, randn(m, k));
Ht = max(0.0, randn(n, k));

tol = 1e-4;
maxiter = 100;

for iter = 1 : maxiter
    violation = 0.0;

    WtW = W' * W;
    AtW = A' * W;

    [new_violation, Ht] = fhals_update(Ht, WtW, AtW);
    violation += new_violation;

    Ht ./= max(eps, vecnorm(Ht));

    HHt = Ht' * Ht;
    AHt = A * Ht;

    [new_violation, W] = fhals_update(W, HHt, AHt);
    violation += new_violation;

    disp(violation);

    if iter == 1
        violation_init = violation;
    else
        if violation_init == 0
            break;
        end

        fitchange = violation / violation_init;

        if fitchange <= tol
            break;
        end
    end
end

Htt = Ht'

end %function