classdef CostQuadratic < ICost
    
    properties
        hyperparams = struct('Costfunction', 'Quadratic')
        costder_limit = 1
    end
    
    methods
        function cost = cost(obj, a, y)
            cost = 0.5 * sum((a-y).^2, 'all');
        end
        
        function cost_der = derivative(obj, a, y)
            cost_der = a - y;
            
            % Gradient Clipping
            if any(abs(cost_der) > obj.costder_limit)
                cost_der = cost_der * obj.costder_limit/max(abs(cost_der));
            end
        end
    end
end

