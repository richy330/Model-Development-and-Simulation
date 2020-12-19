classdef ActivLinear < IActivation
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    
    methods
        function [activation] = a(obj, z)
            activation = z;    
        end
        
        function [da_dz] = dsigma_dz(obj, z)
        	da_dz = 1;    
        end
    end
end