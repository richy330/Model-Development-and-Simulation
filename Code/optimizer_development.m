train_data = readtable("C:\Users\Richa\Dropbox\Studium\Model Development and Simulation\Data\methanedata.csv");
T_me = train_data.T_me';
P_me = train_data.P_me';
T_train = T_me/max(T_me);
P_train = P_me/max(P_me);

%%
stepsize = 2.1;
epochs = 1000;

nn = Network([1,4,4,1], Sigmoid, CostCrossEntropy, SGDOptimizer);
nn.train(T_train, P_train, stepsize, epochs)


figure
plot(T_train, nn.forward(T_train))
hold on
plot(T_train, P_train)
hold off
legend({"P nn", "P PR"})