classdef OptimizerSGD < IOptimizer & matlab.mixin.Copyable
    
    properties
        hyperparams = struct('Optimizer', 'SGD')
    end

    methods  
        %% Descend
        function descend(obj, parent_layer, eta_m, lambda1, lambda2)
            [dCdW, dCdb] = get_gradient(parent_layer);
            W_reg_corr = get_regularization(parent_layer, lambda1, lambda2);

            parent_layer.W = parent_layer.W - eta_m * (dCdW + W_reg_corr);
            parent_layer.b = parent_layer.b - eta_m * dCdb;
        end
    end
end