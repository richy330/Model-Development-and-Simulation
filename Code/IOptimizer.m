classdef IOptimizer < handle
    
    % Abstract Class providing Optimizer-Interface
    properties(Abstract)
        hyperparams
    end
    
    methods(Abstract)
        get_gradient(obj, layer)
        descend(obj, layer)
    end
end

