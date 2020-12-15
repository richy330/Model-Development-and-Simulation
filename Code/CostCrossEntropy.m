classdef CostCrossEntropy < ICost
    
    methods
        function f_cost = cost(obj, a, y)
            f_cost = -(sum(y*log(a) + (1-y)*log(1-a), 'all'));
        end
        
        function cost_der = derivative(obj, a, y)
            cost_der = (a-y) ./ ((a).*(1-a));
        end
    end
end