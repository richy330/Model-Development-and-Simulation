function [delta_W_reg] = get_regularization(layerToBeRegularized, lambda1, lambda2)
    % return Regularization-correction matrix based on L1 and L2 
    % regularization, indicated by lambda1 and lambda2. Default values for 
    % both regularization-coefficients are 0. Correction-matrix has to be
    % SUBTRACTED from original weight matrix to apply correct correction.
    
    W = layerToBeRegularized.W;
    L1_regularization = W*lambda1;

    W_pos = W > 0;
    L2_regularization = zeros(size(W));
    L2_regularization(W_pos) = lambda2;
    L2_regularization(~W_pos) = -lambda2;

    delta_W_reg = L1_regularization + L2_regularization;
end

