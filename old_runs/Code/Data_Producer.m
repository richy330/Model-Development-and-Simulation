% Test for Producing and saving Objects
clc 
clear all

%% Loading the respective Substances and Parameter
Methane = Substance("methane");
Ethane = Substance("ethane");
Propane = Substance("propane");

%% Creating the Temperature sets
N_trainingssets = 20000;
T_methane = linspace(0,Methane.Tc*1.2 ,N_trainingssets);
T_ethane = linspace(0,Ethane.Tc*1.2, N_trainingssets);
T_propane = linspace(0, Propane.Tc*1.2, N_trainingssets);


%% Creating the Trainingsdata sets
[T_methane,P_methane, V_methane] = Peng_Robinson_vectorized(Methane, T_methane, 101325);
[T_ethane,P_ethane, V_ethane] = Peng_Robinson_vectorized(Ethane, T_ethane, 101325);
[T_propane,P_propane, V_propane] = Peng_Robinson_vectorized(Propane, T_propane, 101325);

T_methane = T_Methane( ~isreal(T_methane(:)))

P_crit = []
V_crit = []


%% Save the Produced Data
Data_1 = table(T_methane, P_methane, V_methane, 'VariableNames', {'T_methane', 'P_methane', 'V_methane', } );
Data_2 = table(T_ethane,P_ethane, V_ethan, 'VariableNames', {'T_ethane', 'P_ethane', 'V_ethane', } );
Data_3 = table(T_propane, P_propane, V_propane, 'VariableNames', {'T_propane', 'P_propane', 'V_propane', } );

writetable(Data_1, "PR_TPV_Methane.txt")
writetable(Data_2, "PR_TPV_Ethane.txt")
writetable(Data_3, "PR_TPV_Propane.txt")

save Methane Methane
save Ethane Ethane
save Propane Propane
