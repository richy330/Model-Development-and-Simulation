classdef OptimizerRMSProp < IOptimizer & matlab.mixin.Copyable

    properties
        gamma
        hyperparams = struct('Optimizer', 'RMSProp')
        % cumulative Gradient summation for learningrate-adaption
        state_eta_W = 1
        state_eta_b = 1
    end
    
    methods
        
        %% Constructor
        function obj = OptimizerRMSProp(gamma)
            obj.gamma = gamma;
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
        function descend(obj, parent_layer, eta_m, lambda)            
            [dCdW, dCdb] = get_gradient(parent_layer);
            
            [eta_W, eta_b] = obj.get_adaptive_learningrate(eta_m);
            W_corr = eta_W .* dCdW + eta_m*lambda.*parent_layer.W;
            b_corr = eta_b .* dCdb;
            parent_layer.W = parent_layer.W - W_corr;
            parent_layer.b = parent_layer.b - b_corr;         
            
            obj.update_eta_state(dCdW, dCdb);
        end
    end
end