classdef CostQuadratic < ICost
    
    properties
        hyperparams = struct('Costfunction', 'Quadratic')
    end
    
    methods
        function cost = cost(obj, a, y)
            cost = 0.5 * sum((a-y).^2, 'all');
        end
        
        function cost_der = derivative(obj, a, y)
            cost_der = a - y;
        end
    end
end

