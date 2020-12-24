classdef MonitorPlotter < handle
    
    properties
        plot_directory = pwd
        plot_intervall = 100

        benchmark_xdata
        benchmark_ydata
        x_normalization
        y_normalization
        % added
        plotting_deviation = false
        subplotting = true
        plotting_pressure = false
        plotting_vol = true
        substancedata
        molar_masses
        plot_deviation 
    end
    
    properties(Access = private)
        annotation_pos = [0.15, 0.85, 0, 0]
    end
    
    methods
        %% Constructor
        function obj = MonitorPlotter(benchmark_xdata, benchmark_ydata, substancedata)
            obj.benchmark_xdata = benchmark_xdata;
            obj.benchmark_ydata = benchmark_ydata;
            obj.substancedata = substancedata;
            obj.molar_masses = unique(benchmark_xdata(end,:));
        end
        
        %% Monitor calling Plots
        function monitor(obj, parent_network)
            epoch = parent_network.total_epochs;
            hyperparams = parent_network.hyperparams;
            if mod(epoch, obj.plot_intervall) ~= 0
                return
            end
            
            for substance = 1:numel(obj.molar_masses)

                benchmark_x = obj.benchmark_xdata(:, obj.benchmark_xdata(end,:) == obj.molar_masses(substance));
                benchmark_y = obj.benchmark_ydata(:, obj.benchmark_xdata(end,:) == obj.molar_masses(substance));

                y_nn = parent_network.forward(benchmark_x);

                % if normalizing-coefficients were given, denormalize the
                % plot-data
                if ~isempty(obj.x_normalization)
                    benchmark_x = Normalizer.denormalize(benchmark_x, obj.x_normalization(:,1), obj.x_normalization(:,2));
                end
                if ~isempty(obj.y_normalization)
                    benchmark_y = Normalizer.denormalize(benchmark_y, obj.y_normalization(:,1), obj.y_normalization(:,2));
                    y_nn = Normalizer.denormalize(y_nn, obj.y_normalization(:,1), obj.y_normalization(:,2));
                end
                delta_y_abs = y_nn-benchmark_y;
                delta_y_rel = delta_y_abs ./ benchmark_y * 100;
                
                if obj.subplotting
                    figure('NumberTitle', 'off', 'Name', [obj.substance_identifier(benchmark_x(end,1)) ' Neural Network Behavior'])
                end
                
                if obj.plotting_pressure
                    if obj.subplotting
                        if obj.plotting_vol && obj.plotting_deviation
                            subplot(2,2,1)
                        elseif obj.plotting_vol || obj.plotting_deviation
                            subplot(2,1,1)
                        end
                    end
                    obj.plot_pressure(benchmark_x, benchmark_y, y_nn, hyperparams, epoch);
                end
                if obj.plotting_deviation
                        if obj.plotting_vol && obj.plotting_pressure
                            if obj.subplotting; subplot(2,2,3); end
                            obj.plot_dev_pressure(benchmark_x, delta_y_rel, delta_y_abs, hyperparams, epoch)
                            if obj.subplotting; subplot(2,2,4); end
                            obj.plot_dev_vol(benchmark_x, delta_y_rel, delta_y_abs, hyperparams, epoch)
                        elseif obj.plotting_vol
                            if obj.subplotting; subplot(2,1,2); end
                            obj.plot_dev_vol(benchmark_x, delta_y_rel, delta_y_abs, hyperparams, epoch)
                        else
                            if obj.subplotting; subplot(2,1,2); end
                            obj.plot_dev_pressure(benchmark_x, delta_y_rel, delta_y_abs, hyperparams, epoch)
                        end
                end
                
               
                if obj.plotting_vol 
                    if obj.subplotting
                        if obj.plotting_pressure && obj.plotting_deviation; subplot(2,2,2); 
                        elseif obj.plotting_deviation; subplot(2,2,1);
                        else subplot(2,1,2); end
                    end

                    obj.plot_vol(benchmark_x, benchmark_y, y_nn, hyperparams, epoch)
                end
                 sgtitle([obj.substance_identifier(benchmark_x(end,1)) ' Neural Network Behaviour'])
            end
        end

        
        %% Plotters
        function plot_dev_pressure(obj, x, rel_y_dev, abs_y_dev, hyperparams, epoch)   
            % Plots the 
            if ~obj.subplotting
                figure('NumberTitle', 'off', 'Name', [obj.substance_identifier(x(end,1)) ' Benchmark Plot'])
            end
            
            yyaxis left
            plot(x(1,:), rel_y_dev(1,:))
            ylabel("Deviation [%]")
            
            yyaxis right
            plot(x(1,:), abs_y_dev(1,:))
            ylabel("Deviation [Pa]")
            legend("Deviation [%]", "Deviation [Pa]")
            
            xlabel("Temperature [K]")
            title(['Deviation of Neural Network Prediction'])
            annotation('textbox', 'Position', obj.annotation_pos, 'String', [obj.get_paramstring(hyperparams) 'epochs = ', num2str(epoch)], 'FitBoxToText','on');
        end

        function plot_dev_vol(obj, x, rel_y_dev, abs_y_dev, hyperparams, epoch)           
            if ~obj.subplotting
                figure('NumberTitle', 'off', 'Name', [obj.substance_identifier(x(end,1)) ' Benchmark Plot'])
            end
            
            yyaxis left
            plot(x(1,:), rel_y_dev(2,:), ':', 'LineWidth', 1.5)
            hold on
            plot(x(1,:), rel_y_dev(3,:),  ':', 'LineWidth', 1.5)
            hold off
            ylabel("Deviation [%]")
            
            yyaxis right
            plot(x(1,:), abs_y_dev(2,:), '--', 'LineWidth', 1.5)
            hold on 
            plot(x(1,:), abs_y_dev(3,:), '--', 'LineWidth', 1.5)
            hold off
            ylabel("Deviation [Pa]")
            legend("Deviation_{V_1} [%]" , "Deviation_{V_2} [%]",  "Deviation_{V_1} [Pa]", "Deviation_{V_2} [Pa]")
            
            xlabel("Temperature [K]")
            title(['Deviation of Neural Network Prediction Volume'])
            annotation('textbox', 'Position', obj.annotation_pos, 'String', [obj.get_paramstring(hyperparams) 'epochs = ', num2str(epoch)], 'FitBoxToText','on');
        end
         
        function plot_pressure(obj, x, y_bench, y_nn, hyperparams, epoch)
            
            if ~ obj.subplotting
                figure('NumberTitle', 'off', 'Name', [obj.substance_identifier(x(end,1)) ' Benchmark Plot'])
            end
            plot(x(1,:), y_nn(1,:), 'c')
            hold on
            plot(x(1,:), y_bench(1,:), 'k')
            hold off
            
            legend("P_{nn}", "P_{PR}")
            ylabel("Saturation Pressure [Pa]")
            xlabel("Temperature [K]")
            title('Neural Network Prediction Pressure')
            annotation('textbox', 'Position', obj.annotation_pos, 'String', [obj.get_paramstring(hyperparams) 'epochs = ', num2str(epoch)], 'FitBoxToText','on');
        end 
        
        function plot_vol(obj, x, y_bench, y_nn, hyperparams, epoch)
            if ~ obj.subplotting
                figure('NumberTitle', 'off', 'Name', [obj.substance_identifier(x(end,1)) ' Benchmark Plot'])
            end
            yyaxis right 
            plot(x(1,:), y_nn(2,:), ':', 'LineWidth', 1.5)
            hold on
            plot(x(1,:), y_bench(2,:), 'Color', [0.6350, 0.0780, 0.1840], 'LineStyle', ':', 'LineWidth', 1.5)
            hold off 
            
            yyaxis left 
            plot(x(1,:), y_nn(3,:), 'c--', 'LineWidth', 1.5) 
            hold on
            plot(x(1,:), y_bench(3,:), '--', 'LineWidth', 1.5)
            hold off
            
            legend("V_1_{nn}", "V_1_{PR}", "V_2_{nn}", "V_2_{PR}")
            ylabel("Volume [m^3/mol]")
            xlabel("Temperature [K]")
            title('Neural Network Prediction Volume')
            annotation('textbox', 'Position', obj.annotation_pos, 'String', [obj.get_paramstring(hyperparams) 'epochs = ', num2str(epoch)], 'FitBoxToText','on');
        end 
        
        %% Helpers
        function [paramstring] = get_paramstring(obj, parameter_struct)
            % helper function, creating char-array with name-value pairs from
            % struct
            paramstring = '';
            fns = fieldnames(parameter_struct);
            fvs = struct2cell(parameter_struct);
            for i = 1:numel(fns)
                paramstring = [paramstring fns{i} ' = ' num2str(fvs{i}) newline];
            end
        end
        
        function [substance] = substance_identifier(obj, molar_mass)
            % Through the Input of the Molar mass the Substance is identified
            
            if obj.substancedata.Methane.Mw - molar_mass == 0
                    substance = 'Methane';
            elseif abs(obj.substancedata.Ethane.Mw - molar_mass) < 0.0001
                    substance = 'Ethane';
            elseif abs(obj.substancedata.Propane.Mw - molar_mass) < 0.0001
                    substance = 'Propane';
            elseif abs(obj.substancedata.Butane.Mw - molar_mass) < 0.0001
                    substance = 'Butane';
            elseif abs(obj.substancedata.Pentane.Mw - molar_mass) < 0.0001
                    substance = 'Pentane';
            elseif abs(obj.substancedata.Hexane.Mw - molar_mass) < 0.0001
                    substance = 'Hexane'; 
            else
                    error(['Insert the Molar mass at the last row of the Inputs - the Input ' num2str(molar_mass) ' is not known'])
            end
        end
    end
end


% file_path = ['..\Plots\', 'Epoch', num2str(epoch), '.png'];
% saveas(gcf,file_path);