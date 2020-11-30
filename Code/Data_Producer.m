% Test for Producing and saving Objects
clc 
clear all

%% Loading the respective Substances and Parameter
Methane =  PengRobinson("methane");
Ethane = PengRobinson("ethane");
Propane =  PengRobinson("propane");


%% Creating the Temperature sets
N_trainingssets = 2000;
T_start = 100; 

T_methane = linspace(T_start,Methane.Substance.Tc*0.99,N_trainingssets);
T_ethane = linspace(T_start,Ethane.Substance.Tc*0.99, N_trainingssets);
T_propane = linspace(T_start, Propane.Substance.Tc*0.99, N_trainingssets); 


%% Creating the Trainingsdata sets
Methane.Pressure_Calculation(T_methane);
Ethane.Pressure_Calculation(T_ethane);
Propane.Pressure_Calculation(T_propane);
Methane.Substance.Tc*0.99 %- max(Methane.Ts(:))
Ethane.Substance.Tc*0.99 %- max(Ethane.Ts(:))
Propane.Substance.Tc*0.99 %- max(Propane.Ts(:))

%% Save the Produced Data
save Methane Methane
save Ethane Ethane
save Propane Propane

randi([1,3],1,3)
