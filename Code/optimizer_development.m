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
monitor.plotting_abs_deviation = false;
monitor.plotting_rel_deviation = false;
monitor.plotting_cost = false;
monitor.plot_intervall = 10;



%% Network
beta = 0.9;
gamma = 0.995;
nn = Network([1,100,100,1], ActivReLU, CostQuadratic, OptimizerAdam(beta, gamma));
nn.monitor = monitor;


%% training
learning_rate = 0.01;
epochs = 1000;
lambda = 0.0000000;

tic
nn.train(T_train, P_train, learning_rate, epochs, [], lambda);
toc