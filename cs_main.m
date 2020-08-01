clear

N = 512; %dimensions of x
M = 256;%dimensions of y

global Nit
Nit = 10;
nsim = 5;

mean_pior = rand();
var_pior = 1;
rho = 0.1; %sparsity ratio

snr_db = 20:30;
snr = 10.^(snr_db/10);

nmse1 = zeros(1,length(snr));
nmse2 = zeros(1,length(snr));
nmse3 = zeros(1,length(snr));

for i=1:length(snr)
    for ns = 1:nsim
        A = sqrt(1/M)*randn(M,N);
        x = zeros(N,1);
        for n=1:N
            if rand()<rho
                x(n) = sqrt(var_pior)*randn() + mean_pior;
            end
        end
        
        z = A*x;
        var_noise = norm(z,2)^2./(M*snr(i));
        w = sqrt(var_noise)*randn(M,1);     
        y = z +w;
        
        %% solve with ep
        [x_hat_ep] = cs_with_ep_test(A,y,rho,var_noise,mean_pior,var_pior);
        nmse1(i) =nmse1(i)+ norm(x-x_hat_ep,2)^2/norm(x,2)^2;
        %% solve with L1 minimization
        [x_hat_cvx] = solve_cs_with_L1_minimization(y,A);
        nmse2(i) = nmse2(i)+ norm(x-x_hat_cvx,2)^2/norm(x,2)^2;
        %% solve with AMP
        x_hat_amp = amp_test(A,y,rho,var_noise,mean_pior,var_pior);
        nmse3(i) =nmse3(i)+ norm(x-x_hat_amp,2)^2/norm(x,2)^2;
        
        sim_progress = (ns +nsim*(i-1))/(length(snr)*nsim);
        fprintf("simulation progress:%.3f\n",sim_progress);
    end
end

nmse1 = nmse1/nsim;
nmse2 = nmse2/nsim;
nmse3 = nmse3/nsim;

figure(1)
clf
plot(snr_db,nmse1,'r-')
hold on
plot(snr_db,nmse2,'g-')
hold on
plot(snr_db,nmse3,'b-')
hold on

set(gca,'Yscale','log')
x_min = snr_db(1);
x_max = snr_db(length(snr_db));
y_max = 1;
y_min = 1e-4;

xlabel("snr(dB)")
ylabel("nmse")
legend("CS recovery via EP","CS recovery via L1 minimization","CS recovery via AMP")
axis([x_min x_max y_min y_max])

save("sim_data")












