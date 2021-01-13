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
            adaptive_eta_W = eta_m ./ sqrt(obj.state_W + 1e-4);
            adaptive_eta_b = eta_m ./ sqrt(obj.state_b + 1e-4);
        end
            
        %% Descend
        function descend(obj, parent_layer, eta_m, lambda1, lambda2)
            [eta_W, eta_b] = obj.get_adaptive_learningrate(eta_m);            
            [dCdW, dCdb] = get_gradient(parent_layer);
            W_reg_corr = get_regularization(parent_layer, lambda1, lambda2);

            parent_layer.W = parent_layer.W - (eta_m*W_reg_corr + eta_W.*dCdW);
            parent_layer.b = parent_layer.b - eta_b .* dCdb;
            
            obj.state_W = obj.state_W + dCdW.^2;
            obj.state_b = obj.state_b + dCdb.^2;
        end
    end
end