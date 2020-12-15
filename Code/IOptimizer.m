classdef IOptimizer  
    
    % Abstract Class providing Optimizer-Interface

    
    methods(Abstract)
        get_gradient(obj, layer)
        descend(obj, layer)
    end
end

