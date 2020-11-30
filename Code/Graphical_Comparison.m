
function graphical_comparison(names, Data)
    
    P_max = Data{3}{3}.Substance.Pc;
    T_max = Data{3}{3}.Substance.Tc;
    V_max_1 = max(Data{3}{3}.Vms(1,:));
    V_max_2 = max(Data{3}{3}.Vms(2,:));
    
    for n_inputs = 1:numel(Data)
        
        Input = Data{n_inputs}{1};
        Results = Data{n_inputs}{2};
        PR = Data{n_inputs}{3};

        T = Input(1,:)*T_max;
        P = Results(1,:)*P_max;
        V = Results(2:3,:).*[V_max_1; V_max_2];
        n_varargin = numel(names);
        load(names{1})
        nn_calc = nn.forward(Input);
        
        %% figure comparison
        figure
        % Calculated Data P
        subplot(2,1,1)
        plot(T, P)
        hold on
        plot(PR.Substance.Tc, PR.Substance.Pc,'r*')
        plot(T, nn_calc(1,:)*P_max); 
        hold off
        legend({"PR Data", "Critical Point", names{1:end}}, 'Location', 'southoutside')
        xlabel("Temperature [K]")
        ylabel("Pressure [Pa]")
        title("Comparison of Data")
        
        % Error P
        subplot(2,1,2)
        plot(T, (nn_calc(1,:)-P))
        xlabel("Temperature [K]")
        ylabel("Pressuredifference from PR [Pa]")
        title("Deviation of PR Data [%]" )      
        legend("Deviation from P")
        sgtitle(PR.Substance.name)
        
        
        figure
        % Calculated V1 and V2
        subplot(2,2,1)
        plot(T, V(1,:))
        hold on
        plot(T, nn_calc(2,:)*V_max_1)
        hold off
        xlabel("Temperature [K]")
        ylabel("Volume")
        title("V1 Comparison")
        legend({"PR Data", "NN Data"}, 'Location', 'southoutside')
        
        
        subplot(2,2,2)
        plot(T, V(2,:))
        hold on
        plot(T, nn_calc(3,:)*V_max_2)
        hold off
        title("V2 Comparison")
        xlabel("Temperature [K]")
        ylabel("Volume")
        legend({"PR Data", "NN Data"}, 'Location', 'southoutside')
        
        subplot(2,2,3)
        plot(T, (nn_calc(2,:)*V_max_1-V(1,:)))
        xlabel("Temperature [K]")
        ylabel("Volumedifference from PR [m^3/mol]")
        title("Deviation of PR Data - V1 [%]" )      
        legend("Deviation from V1")
        
        subplot(2,2,4)
        plot(T, (nn_calc(3,:)*V_max_2-V(2,:)))
        xlabel("Temperature [K]")
        ylabel("Volumedifference from PR [m^3/mol]")
        title("Deviation of PR Data - V2 [%]" )      
        legend("Deviation from V2")
        sgtitle(PR.Substance.name)
        
    end
end