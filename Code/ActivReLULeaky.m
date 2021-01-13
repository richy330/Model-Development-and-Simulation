classdef ActivReLULeaky < IActivation    
     
    properties
        hyperparams = struct('Activation', 'Leaky ReLU')
    end
       
    methods
        function [activation] = a(obj, z)
            activation = max(z, 0.1*z);    
        end
        
        function [da_dz] = dsigma_dz(obj, z)
            da_dz = ones(size(z));
        	da_dz(z<0) = 0.1;    
        end
    end
end


