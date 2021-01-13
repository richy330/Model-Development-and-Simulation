% Test for Producing and saving Objects _
clc 
clear all

%% Creating Substances and calculating T-P series
methane = Substance("methane");
ethane = Substance("ethane");
propane = Substance("propane");
butane = Substance("butane");
pentane = Substance("pentane");
hexane = Substance("hexane");

T_range = 80:506;
[Tme, Pme] =  PengRobinson_static.get_vaporpressure(T_range, methane);
[Tet, Pet] =  PengRobinson_static.get_vaporpressure(T_range, ethane);
[Tpr, Ppr] =  PengRobinson_static.get_vaporpressure(T_range, propane);
[Tbu, Pbu] =  PengRobinson_static.get_vaporpressure(T_range, butane);
[Tpe, Ppe] =  PengRobinson_static.get_vaporpressure(T_range, pentane);
[The, Phe] =  PengRobinson_static.get_vaporpressure(T_range, hexane);


%% Plotting
% figure
% plot(Tme, Pme)
% hold on
% plot(Tet,Pet)
% plot(Tpr, Ppr)
% plot(Tbu, Pbu)
% plot(Tpe, Ppe)
% plot(The, Phe)
% hold off
% 
% legend("Methane", "Ethane", "Propane", "Butane", "Pentane", "Hexane")
% xlabel("T [K]")
% ylabel("P [Pa]")

%% Creating supplemental Trainingdata
Mme = methane.Mw*ones(size(Tme));
Met = ethane.Mw*ones(size(Tet));
Mpr = propane.Mw*ones(size(Tpr));
Mbu = butane.Mw*ones(size(Tbu));
Mpe = pentane.Mw*ones(size(Tpe));
Mhe = hexane.Mw*ones(size(The));

ame = methane.acentric_factor*ones(size(Tme));
aet = ethane.acentric_factor*ones(size(Tet));
apr = propane.acentric_factor*ones(size(Tpr));
abu = butane.acentric_factor*ones(size(Tbu));
ape = pentane.acentric_factor*ones(size(Tpe));
ahe = hexane.acentric_factor*ones(size(The));

%% Creating the Trainingsdata sets
x_set = [...
    Tme, Tet, Tpr, The, Tpe, Tbu;
    Mme, Met, Mpr, Mhe, Mpe, Mbu;
    ame, aet, apr, ahe, ape, abu];
y_set = [...
    Pme, Pet, Ppr, Phe, Ppe, Pbu];
numel_train = numel([Tme, Tet, Tpr, The]);

[x_set, x_offset, x_scaling] = Normalizer.autonormalize(x_set, [-4,4]);
[y_set, y_offset, y_scaling] = Normalizer.autonormalize(y_set, [0, 1]);
x_train = x_set(:, 1:numel_train);
y_train = y_set(:, 1:numel_train);
x_but_pent = x_set(:, numel_train+1:end);
y_but_pent = y_set(:, numel_train+1:end);

me_end = numel(Tme); et_end = me_end + numel(Tet); pr_end = et_end + numel(Tpr);
he_end = pr_end + numel(The); pe_end = he_end + numel(Tpe); bu_end = pe_end + numel(Tbu);

plot_delimiter = true(size(y_set));
plot_delimiter([me_end, et_end, pr_end, he_end, pe_end, bu_end]) = false;

%% Monitoring Progress
monitor = MonitorPlotter(x_set, y_set);
monitor.x_normalization = [x_offset, x_scaling];
monitor.y_normalization = [y_offset, y_scaling];
monitor.plot_delimiter = plot_delimiter;

monitor.plotting_benchmark = true;
monitor.plotting_abs_deviation = false;
monitor.plotting_rel_deviation = false;
monitor.plotting_cost = false;
monitor.plot_intervall = 100;


%% Network Setup
beta = 0.5;
gamma = 0.995;
nn = Network([3,250,6,250,1], ActivReLU, CostQuadratic, OptimizerAdam(beta, gamma));
nn.monitor = monitor;


%% Parameter Setup
learning_rate = 0.12;
learning_rate_decay = 0.995;
epochs = 10;
lambda = 0.8;       %L2 Regularization


%% training
while true
tic
nn.train(x_train, y_train, learning_rate, epochs, [], lambda);
toc
learning_rate = learning_rate*learning_rate_decay;
end

