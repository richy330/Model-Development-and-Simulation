function [dCdW, dCdb] = get_gradient(parent_layer)
% Calculating Gradients from a given layer. Scaling Gradients down if they
% exceed a certain limit.
    
    dCdW = [parent_layer.delta * parent_layer.prev.a']';
    dCdb = sum(parent_layer.delta, 2);
    
    % limits clip when norm(gradient) is equal to norm(twos-matrix)
    dCdW_limit = sqrt(2*numel(dCdW));
    dCdb_limit = sqrt(2*numel(dCdb));
    
    
    if norm(dCdW) > dCdW_limit
        dCdW = dCdW * dCdW_limit/norm(dCdW);
    end
    if norm(dCdb) > dCdb_limit
        dCdb = dCdb * dCdb_limit/norm(dCdb);
    end    
end

