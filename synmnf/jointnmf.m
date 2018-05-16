function [W, Htt] = jointnmf(A, S, k)

[m, n] = size(A);

W = max(0.0, randn(m, k));
Ht = max(0.0, randn(n, k));

tempHt = Ht;
left = Ht' * Ht;
right = S * Ht;

tol = 1e-4;
maxiter = 100;

for iter = 1 : maxiter
    violation = 0.0;

    [tempHt, Ht, left, right, new_violation] = symnmf_anls_iter(S, tempHt, Ht, left, right, k);
    violation += new_violation;

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