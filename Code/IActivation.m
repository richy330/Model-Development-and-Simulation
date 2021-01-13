classdef IActivation

    % Abstract Class providing activation-function-Interface
    
    
    methods(Abstract)
        a(obj, z)
        dsigma_dz(obj, z)
    end
    
end
