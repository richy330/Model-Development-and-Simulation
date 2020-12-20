classdef OptimizerAdagrad < IOptimizer & matlab.mixin.Copyable

    properties
        hyperparams = struct('Optimizer', 'Adagrad')
        % cumulative Gradient summation for learningrate-adaption
        state_W = 1
        state_b = 1
    end
    
    methods
        %% Get Adaptive Learningrate
        function [adaptive_eta_W, adaptive_eta_b] = get_adaptive_learningrate(obj, eta_m)
            adaptive_eta_W = eta_m ./ sqrt(obj.state_W);
            adaptive_eta_b = eta_m ./ sqrt(obj.state_b);
        end
        
        %% Get gradient
        function [dCdW, dCdb] = get_gradient(obj, parent_layer)
            dCdW = [parent_layer.delta * parent_layer.prev.a']';
            dCdb = sum(parent_layer.delta, 2);
        end
            
        %% Descend
        function descend(obj, parent_layer, eta_m, lambda)            
            [eta_W, eta_b] = obj.get_adaptive_learningrate(eta_m);            
            [dCdW, dCdb] = obj.get_gradient(parent_layer);

            parent_layer.W = (1 - eta_W*lambda).*parent_layer.W - eta_W .* dCdW;
            parent_layer.b = parent_layer.b - eta_b .* dCdb;
            
            obj.state_W = obj.state_W + dCdW.^2;
            obj.state_b = obj.state_b + dCdb.^2;
        end
    end
end