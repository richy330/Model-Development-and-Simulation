% Model Simulation: Group 3
% Main File

%% Code
%% Loading of Data to run
load Methane.mat
load Ethane.mat
load Propane.mat

plot(Propane.Ts, Propane.Vms)
plot(Methane.Ts, Methane.Vms)
plot(Ethane.Ts, Ethane.Vms)
%% Setup trainingsdata
% Trainingsdata = 1:2:end
% Testdata = 2:2:end

[~, N_coloumns_Methane] = size(Methane.Ts');
[~, N_coloumns_Ethane] = size(Ethane.Ts');
[~, N_coloumns_Propane] = size(Propane.Ts');

x = 10^6;
T_max = Propane.Substance.Tc;
P_max = Propane.Substance.Pc;
V_max_1 = max(Propane.Vms(1,:));
V_max_2 = max(Propane.Vms(2,:));

Input_Methane = [ Methane.Ts'/T_max; repmat([Methane.Substance.antoine_A/x; Methane.Substance.antoine_B/x; Methane.Substance.antoine_C/x; Methane.Substance.Mw], [1, N_coloumns_Methane])];
Input_Ethane = [ Ethane.Ts'/T_max;  repmat([Ethane.Substance.antoine_A/x; Ethane.Substance.antoine_B/x;  Ethane.Substance.antoine_C/x;  Ethane.Substance.Mw],  [1, N_coloumns_Ethane])];
Input_Propane = [ Propane.Ts'/T_max;  repmat([Propane.Substance.antoine_A/x;  Propane.Substance.antoine_B/x;  Propane.Substance.antoine_C/x;  Propane.Substance.Mw],  [1, N_coloumns_Propane])];

Input = [Input_Methane, Input_Ethane, Input_Propane];

Results_Methane = [Methane.Ps'/P_max; (Methane.Vms')./[V_max_1; V_max_2]];

Results_Ethane = [Ethane.Ps'/P_max; (Ethane.Vms')./[V_max_1; V_max_2]];
Results_Propane = [Propane.Ps'/P_max; (Propane.Vms')./[V_max_1; V_max_2]];

Results = [Results_Methane, Results_Ethane, Results_Propane];

Data = {{Input_Methane, Results_Methane, Methane}, {Input_Ethane, Results_Ethane, Ethane}, {Input_Propane, Results_Propane, Propane}};

%% Setting up the NN to train
nn = Network([5,150,3],"sigmoid", "cross-entropy");

disp("--------------------------------NEW RUN--------------------------------")

%load("NN-experiment")

%% Chance of which parameters to use
lambda = 0.0;
stepsize = 15;
limit = 15;
counter = inf;
average_error_prev = 0;
factor = 0.95;

tic
for run = 1:300
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


toc
counter = inf;
average_error_prev = 0;

% Name1 = "NN-from_Scratch_1";
% save(Name1, "nn");


% for i2 = 1:100
%    nn.train(T, P, 32, stepsize, lambda);
%    average_error_new = mean(abs(P - nn.forward(T)));
%    
%    if mod(i2,100) == 0
%        disp(['This is the ' num2str(i2) ' run for lambda!' ])
%        disp(['The Error is ' num2str(average_error_new)])
%    end
%    
%    if average_error_prev < average_error_new
%        if counter == inf
%            counter = i2;
%            average_error_prev = average_error_new;
%            %disp(['average error prev: ' num2str(average_error_prev)])
%        elseif (i2 - counter) > limit
%            lambda = lambda*1.1;
%            counter = inf;
%            average_error_prev = average_error_new;
%            %disp(['Average Error is ', num2str(average_error_new), ' old
%            %error is ' num2str(average_error_prev)])
%            disp(['New lambda is ', num2str(lambda)])
%        end
%    else
%        average_error_prev = average_error_new;
%    end
% end


Name = "NN-experiment-2";
save(Name, "nn");

% Runs that were completed

x = nn.forward(Input);
x = x(2:3, 1:100);
%x = x(3, 1:100);
% Name = Name NN
% Data = Data of trainingsfunctions

Graphical_Comparison({Name}, Data)


