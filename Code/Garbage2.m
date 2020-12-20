% Model Simulation: Group 3 _
% Main File

%% Code
%% Loading of Data to run
load Methane.mat
load Ethane.mat
load Propane.mat
load Butane.mat
load Pentane.mat
load Hexane.mat


%% Setup trainingsdata

[~, N_coloumns_Methane] = size(Methane.Ts');
[~, N_coloumns_Ethane] = size(Ethane.Ts');
[~, N_coloumns_Propane] = size(Propane.Ts');
[~, N_coloumns_Butane] = size(Butane.Ts');
[~, N_coloumns_Pentane] = size(Pentane.Ts');
[~, N_coloumns_Hexane] = size(Hexane.Ts');

% Norm Factors P/V/T
T_max = Propane.Substance.Tc;
P_max = Ethane.Substance.Pc; % ÄNDERUNG ZU ETHANE
V_max_1 = max(Propane.Vms(:,1)); 
V_max_2 = max(Propane.Vms(:,2));

% Norm for Antoine
x_1 = 10;
x_2 = 10^4;
x_3 = 30;

Input_Methane = [ Methane.Ts'/T_max; repmat([Methane.Substance.Mw], [1, N_coloumns_Methane])];
Input_Ethane = [ Ethane.Ts'/T_max;  repmat([Ethane.Substance.Mw],  [1, N_coloumns_Ethane])];
Input_Propane = [ Propane.Ts'/T_max;  repmat([Propane.Substance.Mw],  [1, N_coloumns_Propane])];

Input = [Input_Methane, Input_Ethane, Input_Propane];


Results_Methane = [Methane.Ps'/P_max]; % Methane.Vms'./[V_max_1; V_max_2]];
Results_Ethane = [Ethane.Ps'/P_max]; %Ethane.Vms'./[V_max_1; V_max_2]];
Results_Propane = [Propane.Ps'/P_max]; %Propane.Vms'./[V_max_1; V_max_2]];

Results = [Results_Methane, Results_Ethane, Results_Propane];


Data = {{Input_Methane, Results_Methane, Methane}, {Input_Ethane, Results_Ethane, Ethane}, {Input_Propane, Results_Propane, Propane}};

%% Setting up the NN to train
nn = Network([2,30,40,30,1],"sigmoid", "cross-entropy");

disp("--------------------------------NEW RUN--------------------------------")

load("NN-experiment-3")

%% Chance of which parameters to use
lambda = 0.0;
stepsize = 0.0001;
limit = 15;
counter = inf;
average_error_prev = 0;
factor = 0.95;


for run = 1:1
   nn.train(Input, Results, 32, stepsize);
   average_error_new = mean(sum(abs(Results - nn.forward(Input)),1));
   
   if mod(run,50) == 0
      disp(['RUN: ' num2str(run) ' STEPSIZE: ' num2str(stepsize) ' ERROR: ' num2str(average_error_new) ])
   end

   if average_error_prev < average_error_new
       if counter == inf
           counter = run;
           average_error_prev = average_error_new; 
       elseif (run - counter) > limit
           stepsize = stepsize*factor;
           counter = inf;
           average_error_prev = average_error_new;
       end
   else
       average_error_prev = average_error_new;
   end
end


Name = "NN-experiment-3";
save(Name, "nn");

subplot(3,1,1)
plot(Butane.Ts, Butane.Ps)
hold on
plot(Butane.Ts, P_max*nn.forward([Butane.Ts'/T_max; repmat([ Butane.Substance.Mw], [1, N_coloumns_Butane])]))
hold off
legend({"PR Data", "Prediction Peng Robinson"})
xlabel("Temperature [Pa]")
ylabel("Pressure [K]")
title("Butane")

subplot(3,1,2)
plot(Pentane.Ts, Pentane.Ps)
hold on
plot(Pentane.Ts, P_max*nn.forward([Pentane.Ts'/T_max; repmat([ Pentane.Substance.Mw], [1, N_coloumns_Pentane])]))
hold off
legend({"PR Data", "Prediction Peng Robinson"})
title("Pentane")
xlabel("Temperature [Pa]")
ylabel("Pressure [K]")

subplot(3,1,3)
plot(Hexane.Ts, Hexane.Ps)
hold on
plot(Hexane.Ts, P_max*nn.forward([Hexane.Ts'/T_max; repmat([Hexane.Substance.Mw], [1, N_coloumns_Hexane])]))
hold off
legend({"PR Data", "Prediction Peng Robinson"})
title("Hexane")
xlabel("Temperature [Pa]")
ylabel("Pressure [K]")

% nn.gradient_checking(Input(:,r), Results(:,r))
graphical_comparison_2(Name, Data)
