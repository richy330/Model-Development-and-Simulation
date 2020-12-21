classdef OptimizerSGDMomentum < IOptimizer & handle & matlab.mixin.Copyable
    
    properties
        beta
        dCdW_mom
        dCdb_mom
        hyperparams = struct('Optimizer', 'SGD Momentum')
    end
    
    methods
        %% Constructor
        function obj = OptimizerSGDMomentum(beta)
            obj.dCdW_mom = 0;
            obj.dCdb_mom = 0;
            obj.beta = beta;
            obj.hyperparams.beta = beta;
        end
        
        %% Descend
        function descend(obj, parent_layer, eta_m, lambda)
            [dCdW, dCdb] = get_gradient(parent_layer);
            
            W_corr = obj.beta*obj.dCdW_mom + (1-obj.beta)*eta_m * (dCdW + lambda*parent_layer.W);
            b_corr = obj.beta*obj.dCdb_mom + (1-obj.beta)*eta_m * dCdb;
            parent_layer.W = parent_layer.W - W_corr;
            parent_layer.b = parent_layer.b - b_corr;
            
            obj.dCdW_mom = W_corr;
            obj.dCdb_mom = b_corr;
        end
    end
end