
%% =========================================================================
%  Project    : Proposed Constrained Gradient Descent (CGD) algorithm
%  Author     : Mingxin Deng
%  Affiliation: National University of Defense Technology 
%  Email      : dengmingxin17@gmail.com
%  Created    : [2026-04-13]
%
%  Description:
%  This script implements the constrained gradient descent (CGD) algorithm
%  for compressed sensing recovery.
%  The code is used to reproduce the main numerical results reported in:
%  "Memristive Crossbar Array-based Hardware Framework for Compressed Sensing and Event-Driven Neuromorphic Processing".
%
%
%  Requirements:
%  - MATLAB R2019a
%
%  Citation:
%  If you use this code, please cite the associated paper.
%
%  License:
%  This code is provided for academic and research purposes only.
%% =========================================================================
% y: Compressed signal
% A: Compressiong matrix
% x_ideal: input sparse signal
% plot_flag: plot the details of algorithm
% mul: initial value of variable step size
%
function [x, RSNR,L1_rec,t, t_P, x1, mul_v] = CS_CGD_proposed_auto(y,A,x_ideal,plot_flag,mul)
[m,n]=size(A);

Px = zeros(2,n);

q = A'*inv(A*A')*y;
tic
P = A'*inv(A*A')*A;
t_P = toc;
IP = (eye(n)-P);

z(:,1) = zeros(n,1);
x1(:,1) = q;
x1a(:,1) = zeros(n,1);
I = eye(n);
if nargin < 5
    mul = 0.06; % Default value if initial value is not given
end
mul_v(1:10) = mul;
t = t_P;   
% mul_rec=zeros(1,2);
err_mse = zeros(1,2);
for k=2:61

tic
x1(:,k) = (x1(:,k-1)- mul_v(k)*sign(x1(:,k-1)))  + q - P*(x1(:,k-1)- mul_v(k)*sign(x1(:,k-1)));  % 这公式CPU计算最快
x1(abs(x1(:,k))<0.001,k) = 0;
tk(k) = toc;  % take the time

% Automaticlly shrink the step size
if (mul_v(k)>0.005) && (k>7)&& mean(L1_rec(k-6:k-2))-L1_rec(k-1)< 0.01/1.28
        mul_v(k+1) = mul_v(k) /2;
else
    mul_v(k+1) = mul_v(k);
end

L1_rec(k) = norm(x1(:,k),1);

t = t + tk(k); 

    err = x1(:,k) - x_ideal;
    Rsnr1 = (norm(x_ideal)^2) / (norm(err)^2);  %% RSNR
    RSNR(k) = 10*log10(Rsnr1);
    
    Psnr1 = (max(abs(x_ideal))^2) / (norm(err)^2);
    PSNR(k) = 10*log10((Psnr1)); % PSNR defined in Sci. Adv.
    
    NMSE(k) = 1/Rsnr1;
    
end 
if nargin>3
    if plot_flag==1
        figure
        subplot(4,1,1)
%         axis([0 20 0 10])
        yyaxis left
        RSNR = RSNR';
        ylabel('RSNR')
        plot(RSNR)
        yyaxis right
%         plot(PSNR)
        plot(NMSE)
         ylabel('NMSE')
        title('error trace of x')
        
%         RSNR = norm(x_ideal)/norm((x_ideal)-x1(:,end));
        subplot(4,1,2)
        plot(x_ideal)
        hold on
        plot(x1(:,end))
        title(['恢复结果对比 误差RSNR' num2str(RSNR(end))])
        legend('理想值','恢复值')
        subplot(4,1,3)
        plot(L1_rec)
        title('L1 norm record')
        subplot(4,1,4)
        plot(mul_v)
    end
end


x = x1(:,end);

end
