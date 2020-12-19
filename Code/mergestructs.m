function [merged_struct] = mergestructs(varargin)
% merging given structs and returning new struct containing fields and values
% of both input-structs

    merged_struct = struct;

    for s = 1:numel(varargin)
        struct1 = varargin{s};
        if isempty(struct1)
            continue
        end
        fns = fieldnames(struct1);
        for i = 1:numel(fns)
            fn = fns{i};
            merged_struct.(fn) = struct1.(fn);
        end
    end
end

