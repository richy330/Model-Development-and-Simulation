classdef OptimizerSGD < IOptimizer & matlab.mixin.Copyable
    
    properties
        hyperparams = struct('Optimizer', 'SGD')
    end

    methods
        %% Get gradient
        function [dCdW, dCdb] = get_gradient(obj, parent_layer)
            dCdW = [parent_layer.delta * parent_layer.prev.a']';
            dCdb = sum(parent_layer.delta, 2);
        end
            
        %% Descend
        function descend(obj, parent_layer, eta_m, lambda)
            [dCdW, dCdb] = obj.get_gradient(parent_layer);
            parent_layer.W = (1 - eta_m*lambda)*parent_layer.W - eta_m * dCdW;
            parent_layer.b = parent_layer.b - eta_m * dCdb;
        end
    end
end