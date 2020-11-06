classdef Network < handle
    % Neural Network, consisting of Layer-objects, responsible for training
    % git test
    
    properties
        layers     
        structure
        n_weights
        n_biases
        Cb
    end
    
    methods
        function obj = Network(nn_structure, activ_func, cost_func)
            % Constructs Neural Network, layers with neuron-count
            % determined by nn_structure
            % nn_structure ... vector that defines neurons per layer
            % activ_func   ... string that defines the activation function
            
            if nargin < 2 || isempty(activ_func), activ_func = ""; end       
            if ~isa(activ_func, 'string')
                error("Wrong datatype for variable 'activ_func'. Define activation function by passing a variable of type 'string'")
            end
            if ~isvector(nn_structure) || ~isa(nn_structure, 'double')
                error("Wrong datatype for variable 'nn_structure'. Define Network structure by passing a vector")
            end
            
            obj.structure = nn_structure;
            obj.layers = cell(1, numel(nn_structure));
            
            prev_layer = Layer(nn_structure(1), nn_structure(1), activ_func);
            obj.layers{1} = prev_layer;
            
            obj.n_biases = 0;
            obj.n_weights = 0;
            
            % setting up layers
            for i = 2:numel(nn_structure)
                layer = Layer(nn_structure(i-1), nn_structure(i), activ_func);
                
                prev_layer.next = layer;
                layer.prev = prev_layer;
                obj.layers{i} = layer;
                prev_layer = layer;
                
                obj.n_weights = obj.n_weights + numel(layer.W);
                obj.n_biases = obj.n_biases + numel(layer.b);
            end
            % set proper cos functions for the layers
            switch cost_func
                case "quadratic"
                    f_cost = @(a, y) 0.5 * sum((a-y).^2);
                    f_costDer = @(a, y) a - y;
                case "cross-entropy"
                    f_cost = @(a, y) sum(y*log(a) + (1-y)*log(1-a));
                    f_costDer = @(a, y) (a-y) / ((a)*(1-a));
            end % switch cost_func
                
            for i = 1:numel(obj.layers)
                obj.layers{i}.f_cost = f_cost;
                obj.layers{i}.f_costDer = f_costDer;
            end % setting costfunctions and derivatives for layers
            
        end % Constructor
        
        function [y] = forward(obj, x)
            obj.layers{1}.forward(x);
            y = obj.layers{end}.a;
        end
        
        function backprop(obj, y)
            obj.layers{end}.backprop(y);
        end

        function train(obj, xbatch, ybatch, minibatch_size, stepsize)
            % Run given batch, then update weights and biases according
            % to backpropagation
            % batch ... trainingsdata, 1st row: Input, 2nd row: Result 
            % minibatch_size... size of minibatch, recommended max = 32
            % stepsize ... size of applied adjustment between training sessions
            
            if ~isa(xbatch, 'double')
                error("Wrong datatype of argument 'xbatch' passed to 'train' method in Network. Pass training data as datatype 'double'")
            end
            if ~isa(ybatch, 'double')
                error("Wrong datatype of argument 'ybatch' passed to 'train' method in Network. Pass training data as datatype 'double'")
            end
            if ~isscalar(minibatch_size) || ~(mod(minibatch_size, 1) == 0)
                error("Wrong datatype of argument 'minibatch_size'. Supply minibatch size as integer scalar")
            end
            if ~isscalar(stepsize)
                error("Wrong datatype of argument 'stepsize'. Supply stepsize as scalar")
            end
            
            [mx, nx] = size(xbatch);
            [my, ny] = size(ybatch);
            if ~(mx == obj.layers{1}.n_neurons)
                error("Wrong number of rows in 'xbatch' passed to function 'train'. Row-number should be equal to number of neurons in input-layer")
            elseif ~(my == obj.layers{end}.n_neurons)
                error("Wrong number of rows in 'ybatch' passed to function 'train'. Row-number should be equal to number of neurons in output-layer")
            elseif nx ~= ny
                error("Number of columns in 'xbatch' and 'ybatch' do not agree. Make sure that number of training examples agrees")
            end
            
            % extracting minibatches from given examples,
            % forward x-minibatch to set z and a,
            % backprop y-minibatch to set delta,
            % perform descent based on calculated deltas
            n_minibatches = floor(nx/minibatch_size);
            eta_m = stepsize/minibatch_size;
            for n = 1:n_minibatches
                minibatch_start = (n-1)*minibatch_size + 1;
                minibatch_end = n*minibatch_size;
                
                x_minibatch = xbatch(:, minibatch_start:minibatch_end);
                y_minibatch = ybatch(:, minibatch_start:minibatch_end);
                
                obj.forward(x_minibatch);
                obj.backprop(y_minibatch);
                
                % performing gradient descent on all layers
                for l = 2:numel(obj.layers)
                    obj.layers{l}.descend(eta_m)
                end
                   
            end % processing batch
        end % train

    end % methods
end % classdef

