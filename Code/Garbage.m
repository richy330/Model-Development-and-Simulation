% Model Simulation: Group 3
% Main File

%% Code
%% Loading of Data to run
load Methane.mat
load Ethane.mat
load Propane.mat


%% Setup trainingsdata

[~, N_coloumns_Methane] = size(Methane.Ts');
[~, N_coloumns_Ethane] = size(Ethane.Ts');
[~, N_coloumns_Propane] = size(Propane.Ts');

x = 10^6;
T_max = Propane.Substance.Tc;
P_max = Propane.Substance.Pc;
V_max_1 = max(Propane.Vms(1,:));
V_max_2 = max(Propane.Vms(2,:));
% Correction for Antoine
x_1 = 10;
x_2 = 10^4;
x_3 = 30;

Input_Methane = [ Methane.Ts'/T_max; repmat([Methane.Substance.antoine_A/x_1; Methane.Substance.antoine_B/x_2; Methane.Substance.antoine_C/x_3; Methane.Substance.Mw], [1, N_coloumns_Methane])];
Input_Ethane = [ Ethane.Ts'/T_max;  repmat([Ethane.Substance.antoine_A/x_1; Ethane.Substance.antoine_B/x_2;  Ethane.Substance.antoine_C/x_3;  Ethane.Substance.Mw],  [1, N_coloumns_Ethane])];
Input_Propane = [ Propane.Ts'/T_max;  repmat([Propane.Substance.antoine_A/x_1;  Propane.Substance.antoine_B/x_2;  Propane.Substance.antoine_C/x_3;  Propane.Substance.Mw],  [1, N_coloumns_Propane])];

Input = [Input_Methane, Input_Ethane, Input_Propane];

Results_Methane = [Methane.Ps'/P_max; Methane.Vms'./[V_max_1; V_max_2]];
Results_Ethane = [Ethane.Ps'/P_max; Ethane.Vms'./[V_max_1; V_max_2]];
Results_Propane = [Propane.Ps'/P_max; Propane.Vms'./[V_max_1; V_max_2]];

Results = [Results_Methane, Results_Ethane, Results_Propane];

Data = {{Input_Methane, Results_Methane, Methane}, {Input_Ethane, Results_Ethane, Ethane}, {Input_Propane, Results_Propane, Propane}};

%% Setting up the NN to train
nn = Network([2,150,300,150,3],"sigmoid", "cross-entropy");

disp("--------------------------------NEW RUN--------------------------------")

%load("NN-experiment-2")

%% Chance of which parameters to use
lambda = 0.0;
stepsize = 0.1;
limit = 15;
counter = inf;
average_error_prev = 0;
factor = 0.95;

tic
for run = 1:700
   nn.train(Input_Methane, Results_Methane, 32, stepsize);
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
toc



Name = "NN-experiment-2";
save(Name, "nn");



Graphical_Comparison({Name}, Data)


