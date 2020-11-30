%% compare runs

% x = readtable("Peng_Robinson.txt");
% 
% Name_1 = "NN-run_{100}-design-1-10-10-1-stepsize-01";
% Name_2 = "NN-run_{100}-design-1-10-10-1-stepsize-01_rerun1";
Name_3 = "NN-run_{100}-design-1-10-10-1-stepsize-01_rerun3";
Name_4 = "NN-run_{100}-design-1-10-10-1-stepsize-01_rerun4";
Name_5 = "NN-run4-design-1-10-10-1-stepsize-1_optimization100";
Name_6 = "NN-run4-design-1-10-10-1-stepsize-1_optimization150";
Name_7 = "NN-design_1_10_1";
Name_8 = "NN-design-1-40-1-reruns";
Name_9 = "NN_opt";
Name_10 = "NN-tryout";
Name_11 = "NN-Name_8_L2";

graphical_comparison({Name_6, Name_8 }, x)


function graphical_comparison(names, PR_table)
    T_start = PR_table{:,1}';
    T = T_start/max(T_start);
    P_start = PR_table{:,2}';
    P = P_start/max(P_start);
    P_max = max(P_start);
    T_max = max(T_start);
    
    n_varargin = numel(names);
    load(names{1})
    
    %% figure comparison
    y = nn.forward(T)*P_max;
    y_vector =[];
    figure
    plot(T_start, P_start)
    hold on
    for i = 1:n_varargin
        load(names{i}) 
        y = nn.forward(T)*P_max;
        y_vector = [y_vector;y];
        plot(T_start, y) 
    end
    hold off
    legend({"PR Data", names{1:end}}, 'Location', 'southoutside')
    xlabel("Temperature [K]")
    ylabel("Pressure [Pa]")
    title("Comparison of Data")
    
    %% figure for difference
    figure
    plot(T_start, (y_vector(1,:)-P_start))
    hold on
    for i2 = 2:n_varargin
        plot(T_start, (y_vector(i2,:)-P_start))
    end
    hold off
    legend({names{1:end}}, 'Location', 'southoutside')
    xlabel("Temperature [K]")
    ylabel("Pressuredifference from PR [Pa]")
    title("Deviation of PR Data")
end