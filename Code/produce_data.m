% Test for Producing and saving Objects _
clc 
clear all

%% SubstanceDatas
Substancedata.Methane =  Substance("methane");
Substancedata.Ethane = Substance("ethane");
Substancedata.Propane =  Substance("propane"); 
Substancedata.Butane =  Substance("butane");
Substancedata.Pentane = Substance("pentane");
Substancedata.Hexane =  Substance("hexane"); 


%% Loading the respective Substances and Parameter
PRdata.Methane =  PengRobinson("methane");
PRdata.Ethane = PengRobinson("ethane");
PRdata.Propane =  PengRobinson("propane"); 
PRdata.Butane =  PengRobinson("butane");
PRdata.Pentane = PengRobinson("pentane");
PRdata.Hexane =  PengRobinson("hexane"); 


%% Creating the Temperature sets
N_trainingssets = 500; 

T_methane = linspace(PRdata.Methane.Substance.Tc - 100, PRdata.Methane.Substance.Tc, N_trainingssets);
T_ethane = linspace(PRdata.Ethane.Substance.Tc - 100,PRdata.Ethane.Substance.Tc, N_trainingssets);
T_propane = linspace(PRdata.Propane.Substance.Tc - 100, PRdata.Propane.Substance.Tc, N_trainingssets); 
T_butane = linspace(PRdata.Butane.Substance.Tc - 100, PRdata.Butane.Substance.Tc, N_trainingssets);
T_pentane = linspace(PRdata.Pentane.Substance.Tc - 100, PRdata.Pentane.Substance.Tc, N_trainingssets);
T_hexane = linspace(PRdata.Hexane.Substance.Tc - 100, PRdata.Hexane.Substance.Tc, N_trainingssets); 


%% Creating the Trainingsdata sets
PRdata.Methane.Pressure_Calculation(T_methane);
PRdata.Ethane.Pressure_Calculation(T_ethane);
PRdata.Propane.Pressure_Calculation(T_propane);
PRdata.Butane.Pressure_Calculation(T_butane);
PRdata.Pentane.Pressure_Calculation(T_pentane);
PRdata.Hexane.Pressure_Calculation(T_hexane);


%% Save the Produced Data
save PRdata PRdata
save Substancedata Substancedata
