function [x_hat] = cs_with_ep_test(A,y,rho,var_noise,mean_pior,var_pior)

[M,N] = size(A);
global Nit
epslion =1e-20;

rho_esti = rho;
mean_ai = zeros(M,N);  %fn to vn
var_ai = ones(M,N);

mean_ia = zeros(M,N);  %vn to fn
var_ia = ones(M,N);

Mean_i = zeros(N,1);
Var_i = zeros(N,1);

for iter=1:Nit
    for a=1:M
        for i=1:N
            Zai =0;
            Vai =0;
            for j=1:N
                if i~=j
                    Zai =Zai +A(a,j)*mean_ia(a,j);
                    Vai =Vai +A(a,j)^2*var_ia(a,j);
                end
            end
            mean_ai(a,i) = (y(a)-Zai)/A(a,i);  %fn to vn
            var_ai(a,i) = (var_noise + Vai)/A(a,i)^2;
        end
    end
    
    % attempt to estimate xi
    for i=1:N
        Var_i_ext =inf;
        Mean_i_ext =0;
        for a=1:M            
            [Mean_i_ext,Var_i_ext] = gaussian_mul(Mean_i_ext,Var_i_ext,mean_ai(a,i),var_ai(a,i));
        end

        Z1 = (1-rho_esti)*normpdf(0,Mean_i_ext,sqrt(Var_i_ext + epslion));
        Z2 = rho_esti*normpdf(Mean_i_ext,mean_pior,sqrt(var_pior + Var_i_ext));
        Z =Z1+Z2;  %normlize factor
        
        Mean_i(i) = 1/Z * (Z1 * epslion*Mean_i_ext/(epslion + Var_i_ext) +Z2*(Mean_i_ext*var_pior + mean_pior*Var_i_ext)/(Var_i_ext +var_pior));
        Var_i(i) = Z1/Z *(Var_i_ext*epslion/(Var_i_ext +epslion) + (epslion*Mean_i_ext/(epslion + Var_i_ext) -Mean_i(i))^2) +.....
            Z2/Z *(Var_i_ext*var_pior/(Var_i_ext +var_pior) + ((Mean_i_ext*var_pior + mean_pior*Var_i_ext)/(Var_i_ext +var_pior) -Mean_i(i))^2 );
        
    end
    %vn update
    for i=1:N
        for a=1:M
            [mean_ia(a,i),var_ia(a,i)] = gaussian_div(Mean_i(i),Var_i(i),mean_ai(a,i),var_ai(a,i));
        end
    end
end
x_hat = Mean_i;

end


function [m_new,v_new] = gaussian_mul(m1,v1,m2,v2)
   v_new = 1/(1/v1 +1/v2);
   m_new = v_new *(m1/v1 +m2/v2);
end

function [m_new,v_new] = gaussian_div(m1,v1,m2,v2)
   v_new = 1/(1/v1 -1/v2);
   m_new = v_new*(m1/v1 -m2/v2);  
end