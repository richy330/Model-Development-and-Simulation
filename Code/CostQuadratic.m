classdef CostQuadratic < ICost
    
    methods
        function cost = cost(obj, a, y)
            cost = 0.5 * sum((a-y).^2, 1);
        end
        
        function cost_der = derivative(obj, a, y)
            cost_der = a - y;
        end
    end
end

