classdef Sigmoid < IActivation
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    
    methods
        function [activation] = a(obj, z)
            activation = 1 ./ (1+exp(-z));    
        end
        
        function [da_dz] = dsigma_dz(obj, z)
        	da_dz = (exp(z) ./ (1+exp(z)).^2);    
        end
    end
end

