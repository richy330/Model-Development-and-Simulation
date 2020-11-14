%% Test for Producing and saving Objects

%Methane = Substance("methane");

%T = linspace(0,Methane.Tc ,100000);
%[T,P] = Peng_Robinson_vectorized(Methane, T, 101325);

%save Methane Methane
load NN.mat
nn
%Data = table(T, P,  'VariableNames', { 'Methanol_T', 'Methanol_P'} );
%writetable(Data, "Peng_Robinson.txt")