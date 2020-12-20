% _
function graphical_comparison_2(names, Data)
    
    P_max = Data{3}{3}.Substance.Pc;
    T_max = Data{3}{3}.Substance.Tc;
    
    for n_inputs = 1:numel(Data)
        
        Input = Data{n_inputs}{1};
        Results = Data{n_inputs}{2};
        PR = Data{n_inputs}{3};

        T = Input(1,:)*T_max;
        P = Results(1,:)*P_max;
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
        
        
    end
end