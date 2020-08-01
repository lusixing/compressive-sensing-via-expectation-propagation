function [x_hat] = amp_test(A,y,rho,var_noise,mean_pior,var_pior)

global Nit;

%init
[M,N] = size(A);
v =zeros(M,1);
r = sqrt(var_noise)*randn(N,1);
gamma = var_noise^-1;

x_hat = zeros(N,1);

for iter=1:Nit
    %step1
    for n=1:N
        x_hat(n) = g1_r_gamma(r(n),gamma,mean_pior,var_pior,rho);
    end
    %step2
    a_temp = zeros(N,1);
    for n=1:N
        a_temp(n) = g1_r_gamma_div(r(n),gamma,mean_pior,var_pior,rho);
    end
    a = sum(a_temp)/N;
    %step3
    v = y - A*x_hat + N/M*a*v;
    %step4
    r = x_hat + A'*v;
    %step5
    gamma = M/norm(v,2)^2;
    
end
end

function f1 = g1_r_gamma(r,gamma,mean_p,var_p,rho)
Z1 = sqrt(2*pi/gamma)*( (1-rho)*normpdf(0,r,sqrt(1/gamma)) + rho*normpdf(r,mean_p,sqrt(1/gamma +var_p)) );

v2_temp = var_p/(gamma*var_p+1);
m2_temp = v2_temp*(r*gamma + mean_p/var_p);

f1 = 1/Z1 * sqrt(2*pi/gamma)*(rho * normpdf(r,mean_p,sqrt(1/gamma+var_p))*m2_temp);

end


function f2 = g1_r_gamma_div(r,gamma,mean_p,var_p,rho)
Z1 = sqrt(2*pi/gamma)*( (1-rho)*normpdf(0,r,sqrt(1/gamma)) + rho*normpdf(r,mean_p,sqrt(1/gamma +var_p)) );

v2_temp = var_p/(gamma*var_p+1);
m2_temp = v2_temp*(r*gamma + mean_p/var_p);

Ex = 1/Z1 * sqrt(2*pi/gamma)*(rho * normpdf(r,mean_p,sqrt(1/gamma+var_p))*m2_temp);
Ex2 = 1/Z1 * sqrt(2*pi/gamma)*(rho * normpdf(r,mean_p,sqrt(1/gamma+var_p))*(m2_temp^2+v2_temp));

Dx = Ex2 - Ex^2;

f2 = gamma*Dx;
end


    