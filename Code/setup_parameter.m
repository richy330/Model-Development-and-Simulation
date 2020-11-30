% Model Simulation: Group 3
% Function - setup parameter

function [optimal_stepsize, optimal_lambda] = setup_parameter(stepsize_input, lambda_input, n_runs_input, T, P, batchsize, structure, activation, cost, name)

%% Function Purpose:
%  runs with the given Parameter ranges a certain amount of runs to
%  determinte optimal starting Conditions for the Network to be trained
%% Function Inputs:
%   - stepsize_input ... stepsize vector to be tested
%   - lambda_input   ... lambda vector to be tested
%   - n_runs_input   ... number of runs 
%   - T, P           ... normalized T and P vector
%   - structure      ... structure vector of NN
%   - activation     ... string that names activation function 
%   - cost           ... string that names cost function 
%% Function Output
%   - optimal_stepsize ... best stepsize to start with
%   - optimal_lambda   ... best lambda to start with


%% Initialization of Data
    lowest_average_error = inf;
    average_error = 0;
    optimal_stepsize = 0;
    optimal_lambda = 0;

%% Loop to dedicate the optimum
        for stepsize = stepsize_input
            for lambda = lambda_input
                try
                    network = Network(structure, activation, cost);
                    for n_runs = 1:n_runs_input
                        network.train(T,P,n_runs_input, stepsize, lambda);
                        average_error = mean(abs(P - network.forward(T)));
                        if lowest_average_error > average_error
                            optimal_stepsize = stepsize;
                            optimal_lambda = lambda;
                            lowest_average_error = average_error;
                             disp(['stepsize: ' num2str(stepsize) ' lambda: ' num2str(lambda)])
                        end
                    end
                catch 
                    disp(['The combination of stepsize: ' num2str(stepsize) ' and ' num2str(lambda) ' produces an error!'])
                end
            end             
        end % stepsize
        
end
