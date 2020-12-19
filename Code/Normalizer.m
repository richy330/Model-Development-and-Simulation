classdef Normalizer
    % Helper functions for scaling and offsetting Dataseries to meet limits

    
    methods (Static)
        function [normalized, offset, scaling] = autonormalize(inputsets, limits)
            % Scaling and offsetting input-matrix so that it lies within
            % the given limits. Returns a new matrix of simillar shape, the
            % offset applied to the scaled input in order to center it
            % around the limits, and the scaling factor applied to the
            % inputset to adjust the range of the input to the range of the
            % limits
            if nargin < 2 || isempty(limits), limits = [0, 1]; end
            
            lim_max = max(limits, [], 2);
            lim_min = min(limits, [], 2);
            inp_max = max(inputsets, [], 2);
            inp_min = min(inputsets, [], 2);
            
            scaling = (lim_max-lim_min) ./ (inp_max-inp_min);
            scaled = inputsets .* scaling;
            offset = min(scaled, [], 2) - lim_min;
            normalized = scaled - offset;
        end
        
        function [normalized] = normalize(inputset, offset, scaling)
            % Scaling and offsetting inputmatrix by given parameters.
            % Useful when parameters retrieved from autonormalize need to
            % be applied to other datasets
            normalized = inputset .* scaling - offset;
        end
        
        function [denormalized] = denormalize(normalized, offset, scaling)
            scaled = normalized + offset;
            denormalized = scaled ./ scaling;
        end
    end
end

