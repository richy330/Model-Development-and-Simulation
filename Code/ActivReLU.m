classdef ActivReLU < IActivation
       
    properties
        hyperparams = struct('Activation', 'ReLU')
    end
    
     
    methods
        function [activation] = a(obj, z)
            activation = max(z, 0);    
        end
        
        function [da_dz] = dsigma_dz(obj, z)
        	da_dz = double(z>0);    
        end
    end
end


