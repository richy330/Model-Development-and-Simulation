classdef OptimizerAdam < IOptimizer & matlab.mixin.Copyable

    properties
        gamma
        beta
        hyperparams = struct('Optimizer', 'Adam')
        % cumulative Gradient summation for learningrate-adaption
        state_eta_W = 1
        state_eta_b = 1
        dCdW_mom = 0
        dCdb_mom = 0
    end
    
    methods
        
        %% Constructor
        function obj = OptimizerAdam(beta, gamma)
            obj.beta = beta;
            obj.gamma = gamma;
            obj.hyperparams.beta = beta;
            obj.hyperparams.gamma = gamma;
        end      
        
        %% Get Adaptive Learningrate
        function [adaptive_eta_W, adaptive_eta_b] = get_adaptive_learningrate(obj, eta_m)
            adaptive_eta_W = eta_m ./ sqrt(obj.state_eta_W + 1e-4);
            adaptive_eta_b = eta_m ./ sqrt(obj.state_eta_b + 1e-4);
        end
        
        %% Update Eta-state
        function update_eta_state(obj, dCdW, dCdb)
            obj.state_eta_W = obj.gamma.*obj.state_eta_W + (1-obj.gamma).*dCdW.^2;
            obj.state_eta_b = obj.gamma.*obj.state_eta_b + (1-obj.gamma).*dCdb.^2;
        end
            
        %% Descend
        function descend(obj, parent_layer, eta_m, lambda1, lambda2)            
            [dCdW, dCdb] = get_gradient(parent_layer);
            [eta_W, eta_b] = obj.get_adaptive_learningrate(eta_m);            
            W_reg_corr = get_regularization(parent_layer, lambda1, lambda2);
            W_reg_corr = W_reg_corr * eta_m;
            
            W_grad_corr = obj.beta*obj.dCdW_mom + (1-obj.beta)*eta_W.*dCdW;
            b_grad_corr = obj.beta*obj.dCdb_mom + (1-obj.beta)*eta_b.*dCdb;
            
            parent_layer.W = parent_layer.W - W_grad_corr - W_reg_corr;
            parent_layer.b = parent_layer.b - b_grad_corr;         
            
            obj.dCdW_mom = W_grad_corr;
            obj.dCdb_mom = b_grad_corr;
            obj.update_eta_state(dCdW, dCdb);
        end
    end
end