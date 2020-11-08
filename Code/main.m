% Model Simulation: Group 3
% Main File
sum([1; 2; 3])
nn = Network([1,4,5,1],"sigmoid","cross-entropy");
%nn.train([0.5, 0.5, 0.6], [1,2,3], 1, 0.01)
%nn.train(3, 4, 1, 0.01)

[dC_dW_backprob, dC_db_backprob, dCdW_linear, dCdB_linear] = nn.gradient_checker(3,4);

dC_db_backprob'
dCdB_linear