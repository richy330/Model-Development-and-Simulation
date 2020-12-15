classdef SGDOptimizer < IOptimizer

    methods
        %% Get gradient
        function [dCdW, dCdb] = get_gradient(obj, parent_layer)
            dCdW = [parent_layer.delta * parent_layer.prev.a']';
            dCdb = sum(parent_layer.delta, 2);
        end % get gradient
            
        %% Descend
        function descend(obj, parent_layer, eta_m, lambda)
            [dCdW, dCdb] = obj.get_gradient(parent_layer);
            
            %obj.W = (1 - eta_m*lambda)*obj.W - eta_m * dCdW; %% L2 regularization
            parent_layer.W = parent_layer.W - eta_m * dCdW; %% no regularization
            parent_layer.b = parent_layer.b - eta_m * dCdb;
        end % gradient descent
    end
end

