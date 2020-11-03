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
        function obj = Network(nn_structure, activ_func)
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
            
        end % Constructor
        
        function [y] = forward(obj, x)
            obj.layers{1}.forward(x);
            y = obj.layers{end}.a;
        end
        
        function backprop(obj, x, y)
            % Applying backpropagation algorithm, determining errors for
            % each individual layer and storing them in respective layer
            
            % For sake of single-responibility principle, the gradient 
            % calculation will be implemented in train-function            

            if ~isa(x, 'double')
                error("Wrong datatype for variable 'x' calling backprop. Define input-vector by passing a variable of type 'double'")
            end
            if ~(numel(x) == obj.layers{1}.n_neurons)
                error("Error calling function 'backprop'. Number of elements in passed x-vector is different from neuron-count in first neuron layer")
            end
                
            if ~isa(y, 'double')
                error("Wrong datatype for variable 'y' calling backprop. Define output-vector by passing a variable of type 'double'")
            end
            if ~(numel(y) == obj.layers{end}.n_neurons)
                error("Error calling function 'backprop'. Number of elements in passed y-vector is different from neuron-count in output neuron layer")
            end
            
            
            % forwarding provided data through network, 
            % calculating error-vector for each layer by backpropagation
            obj.layers{1}.forward(x);
            l = obj.layers{end};
            l.delta = (l.a - y) .* l.dsigma_dz;
            W = l.W;
            
            for i = numel(obj.layers)-1:-1:2
                l = obj.layers{i};
                l.delta = (W * l.delta) .* l.dsigma_dz;
                
                W = l.W;                
            end
            
        end % backpropagation
        
        % START NEW IMPLEMENTATION
        function gradient_checking(obj, y)
            cost_function = @(a,y) (y*ln(a)+(1-y)*ln(1-a));
            a_NN = obj.forward(y);
            Cost = cost_function(a_NN, 

        end
        % END NEW IMPLEMENTATION

        function train(obj, batch, minibatch_size, stepsize)
            % Run given batch, then update weights and biases according
            % to backpropagation
            % batch ... trainingsdata, 1st row: Input, 2nd row: Result 
            % minibatch_size... size of minibatch, recommended max = 32
            % stepsize ... size of applied adjustment between training sessions
            
            if ~isa(batch, 'double')
                error("Wrong datatype of argument 'batch' passed to 'train' method in Network. Pass training data as datatype 'double'")
            end
            if ~isscalar(minibatch_size) || ~(mod(minibatch_size, 1) == 0)
                error("Wrong datatype of argument 'minibatch_size'. Supply minibatch size as integer scalar")
            end
            if ~isscalar(stepsize)
                error("Wrong datatype of argument 'stepsize'. Supply stepsize as scalar")
            end
            
            [n_examples, xy] = size(batch);
            if ~(xy == obj.layers{1}.n_neurons + obj.layers{end}.n_neurons)
                error("Wrong number of columns of 'batch' passed to function 'train'. Column number should be equal to number of neurons in input + output layer")
            end
            
            % iterating over examples and extracting minibatches,
            % train on minibatch and adjust weights and biases for each
            % minibatch
            n_minibatches = floor(n_examples/minibatch_size);
            eta_m = stepsize/minibatch_size;
            for n = 1:n_minibatches
                minibatch_start = (n-1)*minibatch_size + 1;
                minibatch_end = n*minibatch_size;
                minibatch = batch(minibatch_start:minibatch_end, :);
                
                bx_corr_matrx = zeros(obj.n_biases, n_minibatches);
                wx_corr_matrx = zeros(obj.n_weights, n_minibatches);
                Cb = 0;
                % iterating over minibatch, applying correction matrices
                % and vectors
                for i = 1:minibatch_size
                    % extract x and y vector from input, set deltas by
                    % backprop-method
                    x = minibatch(i, 1:obj.layers{1}.n_neurons);
                    y = minibatch(i, obj.layers{1}.n_neurons+1:end);
                    obj.backprop(x, y);
                    
                    bx_corr = zeros(1, obj.n_biases);   % col-vector, contains correction values for minibatch
                    wx_corr = zeros(1, obj.n_weights);  % same
                    bx_startindex = 1;
                    wx_startindex = 1;
                    Cb = Cb + 1/2 * (obj.layers{end}.a - y)^2;
                    
                    % iterating through layers, collecting correction-values
                    % of weights and biases, combining them in two vectors
                    for l = 2:numel(obj.layers)
                        bx_endindex = bx_startindex + numel(obj.layers{l}.b)-1;
                        wx_endindex = wx_startindex + numel(obj.layers{l}.W)-1;
                        
                        bx_corr(bx_startindex:bx_endindex) = eta_m * obj.layers{l}.delta;
                        wx_corr(wx_startindex:wx_endindex) = reshape(eta_m * obj.layers{l-1}.a*obj.layers{l}.delta', [], 1);
                        
                        bx_startindex = bx_endindex + 1;
                        wx_startindex = wx_endindex + 1; 
                    end % collecting correction values for 1 example
                    
                    bx_corr_matrx(:, i) = bx_corr;
                    wx_corr_matrx(:, i) = wx_corr;
                end % collectin correction values for 1 minibatch
                obj.Cb = [obj.Cb; Cb];
                bx_corr = sum(bx_corr_matrx, 2);
                wx_corr = sum(wx_corr_matrx, 2);
                
                bx_startindex = 1;
                wx_startindex = 1;
                % applying averaged correction values to all layers
                
                %% possible Error - indizes of matrix might not line up correctly
                for l = 2:numel(obj.layers)
                    bx_endindex = bx_startindex + numel(obj.layers{l}.b)-1;
                    wx_endindex = wx_startindex + numel(obj.layers{l}.W)-1;
                    
                    wxl_corr = reshape(wx_corr(wx_startindex:wx_endindex), size(obj.layers{l}.W));
                    bxl_corr = bx_corr(bx_startindex:bx_endindex);
                    obj.layers{l}.b = obj.layers{l}.b - bxl_corr;
                    obj.layers{l}.W = obj.layers{l}.W - wxl_corr;
                    
                    bx_startindex = bx_endindex + 1;
                    wx_startindex = wx_endindex + 1;
                end % applying correction
                  
            end % processing batch
        end % train

    end % methods
end % classdef

