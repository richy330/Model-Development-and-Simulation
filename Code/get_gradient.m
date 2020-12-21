function [dCdW, dCdb] = get_gradient(parent_layer)
% Calculating Gradients from a given layer. Scaling Gradients down if they
% exceed a certain limit.
    gradient_limit = 1;
    
    dCdW = [parent_layer.delta * parent_layer.prev.a']';
    dCdb = sum(parent_layer.delta, 2);
    
    if any(abs(dCdW) > gradient_limit)
        dCdW = dCdW * gradient_limit/max(abs(dCdW));
    end
    if any(abs(dCdb) > gradient_limit)
        dCdb = dCdb * gradient_limit/max(abs(dCdb));
    end    
end

