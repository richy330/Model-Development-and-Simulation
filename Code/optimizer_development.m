%% Get Trainingdata
train_data = readtable("..\Data\methanedata.csv");
T_me = train_data.T_me(37:end)';
P_me = train_data.P_me(37:end)';

[T_train, T_me_offset, T_me_scaling] = Normalizer.autonormalize(T_me, [-4,4]);
[P_train, P_me_offset, P_me_scaling] = Normalizer.autonormalize(P_me, [0, 1]);



%% Monitoring Progress
monitor = MonitorPlotter(T_train(20:end-0), P_train(20:end-0));
monitor.x_normalization = [T_me_offset, T_me_scaling];
monitor.y_normalization = [P_me_offset, P_me_scaling];

monitor.plotting_benchmark = true;
monitor.plotting_abs_deviation = true;
monitor.plotting_rel_deviation = true;
monitor.plotting_cost = false;
monitor.plot_intervall = 4000;



%% Network
beta = 0.8;
nn = Network([1,100,100,1], ActivSigmoid, CostCrossEntropy, OptimizerRMSProp(0.8));
nn.monitor = monitor;


%% training
learning_rate = 0.05;
epochs = 12000;
lambda = 0.00;

tic
nn.train(T_train, P_train, learning_rate, epochs, [], lambda);
toc