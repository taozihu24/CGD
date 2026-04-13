
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
%% ============================= Main Function ============================================
close all
clear all

M = 4;
N = 8;
K = 1;

% Random compressiion matrix
A = [0.48, 0.20, 0.52, 0.16, 1.05, 0.61, 0.64, 0.87;
    0.90, 0.58, 0.29, 0.71, 0.80, 1.00, 0.96, 0.15;
    0.55, 0.26, 0.58, 1.08, 0.90, 0.20, 0.160, 0.32;
    0.90, 0.48, 1.02, 0.52, 0.36, 0.58, 0.64, 0.32];

Unit_U = 0.2;

% Sparse signal, converted to voltage value
x = [0,0,0,0.0717686014055453,0,0,0,0;0,0,0,0,0,0,0,0.141977818589719;-0.0842862380656753,0,0,0,0,0,0,0;0,0,0,0,0,0,0,-0.167025978254433;0,0,0,-0.135551211630819,0,0,0,0;0,0,0,0,0,0,0,-0.169133392862340;0,0,0,0.117513225221895,0,0,0,0;0,0,0,0,0,0,0,0.114437058417294;-0.117181175802581,0,0,0,0,0,0,0;0.166666666666667,0,0,0,0,0,0,0];

y = A*x';


x_rec_all = [];
mul_all = [];
for i=1:size(x,1)
    stopTol = 1e-10000000;
    x_ideal = x(i,:);
    K = 10;
    mul = 0.15*Unit_U;   % The initial value of variable step size
    [x1r(i,:), RSNR,L1_rec,t, t_P, xi_rec, mul_v] = CS_CGD_proposed_auto(y(:,i),A,x_ideal,0,mul);
    x_rec_all = [x_rec_all; xi_rec];
    mul_all = [mul_all; mul_v];
    RSNR_ind(i) = 10*log10(1/MMSE(x1r(i,:),x(i,:)).^2);
    NMSE_ind(i) = norm(x1r(i,:)-x(i,:)).^2 / norm(x(i,:))^2;
end

origin_data = x_rec_all'/Unit_U; 

x_span = reshape(x',1,[]);
x_rec_span = reshape(x1r',1,[]);
for i=1:size(x_rec_all,2)
    NMSE_all(i) = MMSE(x_rec_all(:,i),x_span')^2;
    RSNR_all(i) = 10*log10(1/NMSE_all(i));
end

figure
subplot(2,2,1)
plot(x_span,'--')
hold on
plot(x_rec_span,'*')
title(['Recovery Results, RSNR=' num2str(RSNR_all(end))])

subplot(2,2,2)
plot(RSNR_all)
title('RSNR Trace')
ylabel('RSNR(dB)')
subplot(2,2,3)
hold on
for i=1:size(mul_v,1)
    plot(mul_v(i,:))
end
title('Variable step size')
subplot(2,2,4)
hold on
for i=1:size(x_rec_all,1)
    plot(x_rec_all(i,:))
end
title('Convergence trace of x')




