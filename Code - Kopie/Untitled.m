clc
close all


%% Load P and T Data
P = x{:,2}';
P_max = max(P);
P_extra = P(L_1);
P = [P,P_extra, P_extra ,P_extra, P_extra];
P = P/max(P);
T = T(P > 0.01);
P = P(P > 0.01);
T = T(1:100:end);
P = P(1:100:end);

%% 
stepsize_input = [0.01, 0.1, 1, 1.1, 1.3, 1.5 2, 3, 5, 8, 10];
lambda_input = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];

activation = "sigmoid";
cost = "cross-entropy";
name = "Network_tryout";
batchsize = 32;

[optimal_stepsize, optimal_lambda] = setup_parameter(stepsize_input, lambda_input, n_runs_input, T, P, batchsize, structure, activation, cost, name);