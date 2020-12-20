classdef ActivLinear < IActivation
    
    properties
        hyperparams = struct('Activation', 'Linear')
    end
    
    methods
        function [activation] = a(obj, z)
            activation = z;    
        end
        
        function [da_dz] = dsigma_dz(obj, z)
        	da_dz = 1;    
        end
    end
end