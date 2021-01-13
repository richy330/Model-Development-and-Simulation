classdef ICost

    methods(Abstract)
        cost(obj, a, y)
        derivative(obj, a, y)
    end
end

