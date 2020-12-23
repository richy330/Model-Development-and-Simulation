function [dCdW, dCdb] = get_gradient(parent_layer)
% Calculating Gradients from a given layer. Scaling Gradients down if they
% exceed a certain limit.
    gradient_limit = 10;
    
    dCdW = [parent_layer.delta * parent_layer.prev.a']';
    dCdb = sum(parent_layer.delta, 2);
    
    if norm(dCdW) > gradient_limit
        dCdW = dCdW * gradient_limit/norm(dCdW);
    end
    if norm(dCdb) > gradient_limit
        dCdb = dCdb * gradient_limit/norm(dCdb);
    end    
end

