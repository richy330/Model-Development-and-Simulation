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
        function descend(obj, parent_layer, eta_m, lambda1, lambda2)
            [dCdW, dCdb] = get_gradient(parent_layer);
            W_reg_corr = get_regularization(parent_layer, lambda1, lambda2);
            W_reg_corr = W_reg_corr * eta_m;
            
            W_grad_corr = obj.beta*obj.dCdW_mom + (1-obj.beta)*eta_m * dCdW;
            b_grad_corr = obj.beta*obj.dCdb_mom + (1-obj.beta)*eta_m * dCdb;
            
            parent_layer.W = parent_layer.W - W_grad_corr - W_reg_corr;
            parent_layer.b = parent_layer.b - b_grad_corr;
            
            obj.dCdW_mom = W_grad_corr;
            obj.dCdb_mom = b_grad_corr;
        end
    end
end