classdef MonitorPlotter < handle
    
    properties
        plot_directory = pwd
        plot_intervall = 100
        plotting_rel_deviation = false
        plotting_abs_deviation = false
        plotting_benchmark = false
        plotting_cost = false
        benchmark_xdata
        benchmark_ydata
        plot_delimiter
        x_normalization
        y_normalization
        costs
        epochs
    end
    
    properties(Access = private)
        annotation_pos = [0.15, 0.85, 0, 0]
    end
    
    methods
        %% Constructor
        function obj = MonitorPlotter(benchmark_xdata, benchmark_ydata)
            obj.benchmark_xdata = benchmark_xdata;
            obj.benchmark_ydata = benchmark_ydata;
        end
        
        %% Monitor calling Plots
        function monitor(obj, parent_network)
            epoch = parent_network.total_epochs;
            obj.costs = [obj.costs, parent_network.cost];
            obj.epochs = [obj.epochs, epoch];
            if mod(epoch, obj.plot_intervall) ~= 0
                return
            end

            x_plot = obj.benchmark_xdata(1,:);
            y_plot = obj.benchmark_ydata(1,:);
            x_plot(~obj.plot_delimiter) = nan;
            y_plot(~obj.plot_delimiter) = nan;
            
            y_nn = parent_network.forward(obj.benchmark_xdata);
            % if normalizing-coefficients were given, denormalize the
            % plot-data
            if ~isempty(obj.x_normalization)
                x_plot = Normalizer.denormalize(x_plot, obj.x_normalization(1), obj.x_normalization(2));
            end
            if ~isempty(obj.y_normalization)
                y_plot = Normalizer.denormalize(y_plot, obj.y_normalization(1), obj.y_normalization(2));
                y_nn = Normalizer.denormalize(y_nn, obj.y_normalization(1), obj.y_normalization(2));
            end
%             delta_y_abs = y_nn-benchmark_y;
%             delta_y_rel = delta_y_abs ./ benchmark_y * 100;
            
            hyperparams = parent_network.hyperparams;
            if obj.plotting_rel_deviation
                obj.plot_rel_dev(x_plot, delta_y_rel, hyperparams, epoch);
            end
            if obj.plotting_abs_deviation
                obj.plot_abs_dev(x_plot, delta_y_abs, hyperparams, epoch);
            end
            if obj.plotting_benchmark
                obj.plot_benchmark(x_plot, y_plot, y_nn, hyperparams, epoch);
            end
            if obj.plotting_cost
                obj.plot_cost(hyperparams, epoch)
            end
        end
        
        %% Plotters
        function plot_rel_dev(obj, x, rel_y_dev, hyperparams, epoch)
            figure
            plot(x, rel_y_dev)
            
            legend("(P nn - P PR) / P PR")
            ylabel("Deviation %")
            xlabel("T [K]")
            title(['Relative Deviation Peng Robinson vs Neural Net' newline...
                'epochs=', num2str(epoch)])
            annotation('textbox', 'Position', obj.annotation_pos, 'String', obj.get_paramstring(hyperparams), 'FitBoxToText','on');
            drawnow;
        end
        function plot_abs_dev(obj, x, abs_y_dev, hyperparams, epoch)
            figure
            plot(x, abs_y_dev)
            
            legend("P nn - P PR")
            ylabel("Deviation Pa")
            xlabel("T [K]")
            title(['Total Deviation Peng Robinson vs Neural Net' newline...
                'epochs=', num2str(epoch)])
            annotation('textbox', 'Position', obj.annotation_pos, 'String', obj.get_paramstring(hyperparams), 'FitBoxToText','on');
            drawnow;
        end
        function plot_benchmark(obj, x, y_bench, y_nn, hyperparams, epoch)
            figure
            plot(x, y_nn)
            hold on
            plot(x, y_bench)
            hold off
            
            legend("P nn", "P PR")
            ylabel("Saturation Pressure Pa")
            xlabel("T [K]")
            title(['Peng Robinson vs Neural Net' newline...
                'epochs=', num2str(epoch)])
            annotation('textbox', 'Position', obj.annotation_pos, 'String', obj.get_paramstring(hyperparams), 'FitBoxToText','on');
            drawnow;
        end
        function plot_cost(obj, hyperparams, epoch)
            figure
            plot(obj.epochs, obj.costs)
            
            legend("Cost")
            ylabel("Cost")
            xlabel("Epochs")
            ylim([0,30])
            title(['Peng Robinson vs Neural Net' newline...
                'epochs=', num2str(epoch)])
            annotation('textbox', 'Position', obj.annotation_pos, 'String', obj.get_paramstring(hyperparams), 'FitBoxToText','on');
            drawnow;
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
    end
end


% file_path = ['..\Plots\', 'Epoch', num2str(epoch), '.png'];
% saveas(gcf,file_path);