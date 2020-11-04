% Model Simulation: Group 3
% Main File

nn = Network([1,4,5,1],"sigmoid","quadratic");
nn.train([0.5, 0.5, 0.6], [1,2,3], 1, 0.01)

