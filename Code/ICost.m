classdef ICost
    
    properties(Abstract)
        hyperparams
    end

    methods(Abstract)
        cost(obj, a, y)
        derivative(obj, a, y)
    end
end

