%% compare runs

x = readtable("Peng_Robinson.txt");
T_start = x{:,1}';
T = T_start/max(T_start);
P_start = x{:,2}';
P = P_start/max(P_start);
P_max = max(P_start);
T_max = max(T_start);



load("NN_sig_ent_40_1")
nn_1 = nn;
load NN_sig_ent_400_001
nn_2 = nn;
load NN_quad_ReLU_40_1
nn_3 = nn;
load NN_sigent_runs40_n1_150_1
nn_4 = nn;
load NN_sigent_runs40_n1_150_150_150_1
nn_5 = nn;


T_plot = T*T_max;
y1 = nn_1.forward(T)*P_max;
y2 = nn_2.forward(T)*P_max;
y3 = nn_3.forward(T)*P_max;
y4 = nn_4.forward(T)*P_max;
y5 = nn_5.forward(T)*P_max;

figure
subplot(2,1,1)
pr = plot(T_start,P_start, 'k');
hold on
nn_1_plot = plot(T_plot, y1);
nn_2_plot = plot(T_plot, y2);
nn_3_plot = plot(T_plot, y3);
nn_4_plot = plot(T_plot, y4);
nn_5_plot = plot(T_plot, y5);
hold off
xlabel("Temperature [K]")
ylabel("Pressure [Pa]")
legend({"Original Data","nn_1 sig,ent - 400_{runs} - Stepsize_{0.01}", "nn_2  sig,ent - 400_{runs} - Stepsize_1", "nn_3 ReLU,quad - 400_{runs} Stepsize_1", "run4", "run5"}, 'Location', 'northwest')
subplot(2,1,2)
difference_1 = plot(T_plot, y1 - P_start);
hold on
difference_2 = plot(T_plot, y2 - P_start);
difference_3 = plot(T_plot, y3 - P_start);
difference_4 = plot(T_plot, y4 - P_start);
difference_5 = plot(T_plot, y5 - P_start);
hold off
xlabel("Temperature [K]")
ylabel("Pressuredifference from PR [Pa]")