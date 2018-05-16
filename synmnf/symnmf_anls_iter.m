function [W, H, left, right, violation] = symnmf_anls_iter(A, W, H, left, right, k)

maxiter = 10000;
tol = 1e-3;
alpha = max(max(A))^2;

I_k = alpha * eye(k);

W = nnlsm_blockpivot(left + I_k, (right + alpha * H)', 1, W')';

left = W' * W;
right = A * W;

H = nnlsm_blockpivot(left + I_k, (right + alpha * W)', 1, H')';

tempW = sum(W, 2);
tempH = sum(H, 2);
temp = alpha * (H-W);

gradH = H * left - right + temp;

left = H' * H;
right = A * H;

gradW = W * left - right - temp;

violation = sqrt(norm(gradW(gradW<=0|W>0))^2 + norm(gradH(gradH<=0|H>0))^2);

end