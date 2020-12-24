%% Get Trainingdata
% Model Simulation: Group 3 _
% Main File

%% Code
%% Loading of Data to run
clc
close all

load PRdata.mat
load Substancedata.mat
Methane = PRdata.Methane;
Ethane = PRdata.Ethane;
Propane = PRdata.Propane;


%% Setup trainingsdata

[~, N_coloumns_Methane] = size(Methane.Ts');
[~, N_coloumns_Ethane] = size(Ethane.Ts');
[~, N_coloumns_Propane] = size(Propane.Ts');

Input_Methane = [ Methane.Ts'; repmat([Methane.Substance.antoine_A; Methane.Substance.antoine_B; Methane.Substance.antoine_C; Methane.Substance.Mw], [1, N_coloumns_Methane])];
Input_Ethane = [ Ethane.Ts';  repmat([Ethane.Substance.antoine_A; Ethane.Substance.antoine_B;  Ethane.Substance.antoine_C;  Ethane.Substance.Mw],  [1, N_coloumns_Ethane])];
Input_Propane = [ Propane.Ts';  repmat([Propane.Substance.antoine_A;  Propane.Substance.antoine_B;  Propane.Substance.antoine_C;  Propane.Substance.Mw],  [1, N_coloumns_Propane])];

Input = [Input_Methane, Input_Ethane, Input_Propane];

Results_Methane = [Methane.Ps'; Methane.Vms'];
Results_Ethane = [Ethane.Ps'; Ethane.Vms'];
Results_Propane = [Propane.Ps'; Propane.Vms'];

Results = [Results_Methane, Results_Ethane, Results_Propane];
Data = {{Input_Methane, Results_Methane, Methane}, {Input_Ethane, Results_Ethane, Ethane}, {Input_Propane, Results_Propane, Propane}};


%% Normalizing of Data
[Input_train, Input_offset, Input_scaling] = Normalizer.autonormalize(Input, [-4,4; 0,1; 0,1; 0,1; 0,1]);
[Results_train, Results_offset, Results_scaling] = Normalizer.autonormalize(Results, [0, 1; 0,1; 0,1]);


%% Monitoring Progress
monitor = MonitorPlotter(Input_train, Results_train, Substancedata);
monitor.x_normalization = [Input_offset, Input_scaling];
monitor.y_normalization = [Results_offset, Results_scaling];

monitor.plotting_pressure = true;
monitor.plotting_deviation = true;
monitor.plot_intervall = 60;


%% Network
beta = 0.8;
nn = Network([5,10,10,3], ActivSigmoid, CostCrossEntropy, OptimizerSGDMomentum(beta));
nn.monitor = monitor; 


%% training
stepsize = 10;
epochs = 60;
lambda = 0.00;

tic
nn.train(Input_train, Results_train, stepsize, epochs, [], lambda);
toc
