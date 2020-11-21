% Model Simulation: Group 3
% Main File

x = readtable("../Data/Peng_Robinson.txt");
T = x{:,1}';
P = x{:,1}';

nn = Network([1,4,5,1],"sigmoid","cross-entropy");
nn.train(T, P, 32, 0.01)

[P_nn] = nn.forward(T)
nn.train(3, 4, 1, 0.01)