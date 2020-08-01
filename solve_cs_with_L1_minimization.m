function [x_hat] = solve_cs_with_L1_minimization(y,A)

[M,N] = size(A);
x_hat = zeros(N,1);
try
cvx_begin quiet
variable x_hat(N,1)
minimize(norm(x_hat,1))
subject to
A*x_hat ==y
cvx_end

catch
   fprintf("cvx not installed, skipped\n") 
end