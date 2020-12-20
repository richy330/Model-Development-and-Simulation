classdef IActivation

    % Abstract Class providing activation-function-Interface
    properties(Abstract)
        hyperparams
    end
    
    methods(Abstract)
        a(obj, z)
        dsigma_dz(obj, z)
    end
    
end
