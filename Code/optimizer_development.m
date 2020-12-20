%% Get Trainingdata
train_data = readtable("..\Data\methanedata.csv");
T_me = train_data.T_me';
P_me = train_data.P_me';

[T_train, T_me_offset, T_me_scaling] = Normalizer.autonormalize(T_me, [-4,4]);
[P_train, P_me_offset, P_me_scaling] = Normalizer.autonormalize(P_me, [0, 1]);



%% Monitoring Progress
monitor = MonitorPlotter(T_train(40:end-0), P_train(40:end-0));
monitor.x_normalization = [T_me_offset, T_me_scaling];
monitor.y_normalization = [P_me_offset, P_me_scaling];

monitor.plotting_benchmark = true;
monitor.plotting_abs_deviation = true;
monitor.plotting_rel_deviation = true;
monitor.plot_intervall = 6000;



%% Network
beta = 0.8;
nn = Network([1,10,10,1], ActivSigmoid, CostCrossEntropy, OptimizerSGDMomentum(beta));
nn.monitor = monitor;


%% training
stepsize = 10;
epochs = 6000;
lambda = 0.00;

tic
nn.train(T_train, P_train, stepsize, epochs, [], lambda);
toc

