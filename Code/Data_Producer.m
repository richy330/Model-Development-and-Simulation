% Test for Producing and saving Objects _
clc 
clear all

%% Loading the respective Substances and Parameter
Methane =  PengRobinson("methane");
Ethane = PengRobinson("ethane");
Propane =  PengRobinson("propane");
Butane = PengRobinson("butane");
Pentane = PengRobinson("pentane");
Hexane = PengRobinson("hexane");


%% Creating the Temperature sets
N_trainingssets = 500;
T_start = 100; 

T_methane = linspace(T_start,Methane.Substance.Tc,N_trainingssets);
T_ethane = linspace(T_start,Ethane.Substance.Tc, N_trainingssets);
T_propane = linspace(T_start, Propane.Substance.Tc, N_trainingssets); 
T_butane = linspace(T_start,Butane.Substance.Tc,N_trainingssets);
T_pentane = linspace(T_start,Pentane.Substance.Tc, N_trainingssets);
T_hexane = linspace(T_start, Hexane.Substance.Tc, N_trainingssets); 


%% Creating the Trainingsdata sets
Methane.Pressure_Calculation(T_methane);
Ethane.Pressure_Calculation(T_ethane);
Propane.Pressure_Calculation(T_propane);
Butane.Pressure_Calculation(T_butane);
Pentane.Pressure_Calculation(T_pentane);
Hexane.Pressure_Calculation(T_hexane);

%% Save the Produced Data
save Methane Methane
save Ethane Ethane
save Propane Propane
save Butane Butane
save Pentane Pentane
save Hexane Hexane