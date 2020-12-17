classdef OptimizerSGDMomentum < IOptimizer & handle
    
    properties
        beta
        layer_mom_map
    end
    
    methods
        %% Constructor
        function obj = OptimizerSGDMomentum(beta)
            obj.beta = beta;
            obj.layer_mom_map = containers.Map('KeyType', 'double', 'ValueType','any');
        end
        
        %% Descend
        function descend(obj, parent_layer, eta_m, lambda)
            [dCdW, dCdb] = obj.get_gradient(parent_layer);
            [dCdW_mom, dCdb_mom] = obj.get_momentum(parent_layer);
            
            W_corr = obj.beta*dCdW_mom + (1-obj.beta)*eta_m * (dCdW + lambda*parent_layer.W);
            b_corr = obj.beta*dCdb_mom + (1-obj.beta)*eta_m * dCdb;
            parent_layer.W = parent_layer.W - W_corr;
            parent_layer.b = parent_layer.b - b_corr;
            
            obj.store_momentum(parent_layer, W_corr, b_corr);
        end
        
        
        %% Get gradient
        function [dCdW, dCdb] = get_gradient(obj, parent_layer)
            dCdW = [parent_layer.delta * parent_layer.prev.a']';
            dCdb = sum(parent_layer.delta, 2);
        end
        
        %% Get Momentum
        function [dCdW, dCdb] = get_momentum(obj, parent_layer)
            map = obj.layer_mom_map;
            uuid = parent_layer.uuid;
            
            if ~isKey(map, uuid)
                momentum = struct;
                momentum.W_corr = zeros(size(parent_layer.W));
                momentum.b_corr = zeros(size(parent_layer.b));
                map(uuid) = momentum;
                obj.layer_mom_map = map;
            end
            dCdW = obj.layer_mom_map(uuid).W_corr;
            dCdb = obj.layer_mom_map(uuid).b_corr;
        end
        
        %% Store Momentum
        function store_momentum(obj, parent_layer, W_corr, b_corr)
            momentum = struct('W_corr', W_corr, 'b_corr', b_corr);
            uuid = parent_layer.uuid;
            obj.layer_mom_map(uuid) = momentum;
        end
    end
end