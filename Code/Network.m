classdef Network < handle
    % Neural Network, consisting of Layer-objects, responsible for training
    % git test _
    
    properties
        layers     
        structure {mustBeInteger, mustBePositive}
        n_weights = 0
        n_biases = 0
        total_epochs = 1
        hyperparams_ = struct()
        hyperparams
        cost
        monitor
    end
    
    methods
        %% Constructor
        function obj = Network(nn_structure, activ_func, cost_func, optimizer)
            % Constructs Neural Network, layers with neuron-count
            % determined by nn_structure
            % nn_structure ... vector that defines neurons per layer
            % activ_func   ... IActiv object, providing activation function
            % cost_func    ... ICost object, providing cost function
            % optimizer    ... IOptimizer object, providing training capabilities
            
            if ~isa(activ_func, 'IActivation')
                error(["Wrong objecttype for 'activ_func'. expected 'IActivation', got ", class(activ_func), " instead"])
            end
            if ~isa(cost_func, 'ICost')
                error(["Wrong objecttype for 'cost_func'. expected 'ICost', got ", class(cost_func), " instead"])
            end
            if ~isa(optimizer, 'IOptimizer')
                error(["Wrong objecttype for 'optimizer'. expected 'IOptimizer', got ", class(optimizer), " instead"])
            end
            if ~isvector(nn_structure) || ~isa(nn_structure, 'double')
                error(["Wrong datatype for variable 'nn_structure'. Define Network structure by passing a double-vector"])
            end
            
            obj.structure = nn_structure;
            obj.hyperparams_.Networkstructure = nn_structure;
            obj.layers = cell(1, numel(nn_structure));
            
            % setting up layers, copy(optimizer) needed to provide unique
            % optimizer for every layer, not a handle to the same optimizer
            prev_layer = Layer(nn_structure(1), nn_structure(1), activ_func, cost_func, copy(optimizer));
            obj.layers{1} = prev_layer;
            
            for i = 2:numel(nn_structure)
                layer = Layer(nn_structure(i-1), nn_structure(i), activ_func, cost_func, copy(optimizer));
                
                prev_layer.next = layer;
                layer.prev = prev_layer;
                obj.layers{i} = layer;
                prev_layer = layer;
                
                obj.n_weights = obj.n_weights + numel(layer.W);
                obj.n_biases = obj.n_biases + numel(layer.b);
            end
            obj.layers{end}.activ_func = ActivLinear;
        end % Constructor
        
        %% Forwarding and Backpropagation
        function [y] = forward(obj, x)
            obj.check_validity(x, "forwarded Values");
            obj.layers{1}.forward(x);
            y = obj.layers{end}.a;
        end
       

        function backprop(obj, y)
            obj.check_validity(y, "backpropagated Results");
            obj.layers{end}.backprop(y);
        end
        
        %% Training
        function train(obj, xbatch, ybatch, learning_rate, epochs, minibatch_size, lambda1, lambda2, randomize)
            % Run given batch, then update weights and biases according
            % to backpropagation
            % xbatch... input-trainingdata, rows=input-neurons, columns=training-examples
            % xbatch... output-trainingdata, rows=ouput-neurons, columns=training-examples            
            % learning_rate ... size of applied adjustment between training-iterations
            % epochs ... number of repeated loops over testdata, default=1            
            % minibatch_size... size of minibatch, default=32
            % lambda1... L1 Regularization coeeficient, default=0
            % lambda2... L2 Regularization coeeficient, default=0
            % randomize... Boolean, training-examples will be shuffled when true, default=True
            if nargin < 9 || isempty(randomize), randomize = true; end
            if nargin < 8 || isempty(lambda2), lambda2 = 0; end
            if nargin < 7 || isempty(lambda1), lambda1 = 0; end
            if nargin < 6 || isempty(minibatch_size), minibatch_size = 32; end
            if nargin < 5 || isempty(epochs), epochs = 1; end
            
            if ~isa(xbatch, 'double')
                error("Wrong datatype of argument 'xbatch' passed to 'train' method in Network. Pass training data as datatype 'double'")
            end
            if ~isa(ybatch, 'double')
                error("Wrong datatype of argument 'ybatch' passed to 'train' method in Network. Pass training data as datatype 'double'")
            end
            obj.check_validity(xbatch, "xbatch");
            obj.check_validity(ybatch, "ybatch");
            
            if ~isscalar(minibatch_size) || ~(mod(minibatch_size, 1) == 0)
                error("Wrong datatype of argument 'minibatch_size'. Supply minibatch size as integer scalar")
            end
            if ~isscalar(learning_rate)
                error("Wrong datatype of argument 'learning_rate'. Supply learning_rate as scalar")
            end
            
            obj.hyperparams_.learning_rate = learning_rate;
            
            
            tic;
            for epoch = 1:epochs
                [mx, nx] = size(xbatch); % columns = trainings examples
                [my, ny] = size(ybatch);
                if ~(mx == obj.layers{1}.n_neurons)
                    error("Wrong number of rows in 'xbatch' passed to function 'train'. Row-number should be equal to number of neurons in input-layer")
                elseif ~(my == obj.layers{end}.n_neurons)
                    error("Wrong number of rows in 'ybatch' passed to function 'train'. Row-number should be equal to number of neurons in output-layer")
                elseif nx ~= ny
                    error("Number of columns in 'xbatch' and 'ybatch' do not agree. Make sure that number of training examples agrees")
                end
                
                if randomize
                    rand_perm = randperm(nx); % random numbers from 1:number of trainingsexamples
                    xbatch = xbatch(:, rand_perm);
                    ybatch = ybatch(:, rand_perm);
                end
                % extracting minibatches from given examples,
                % forward x-minibatch to set z and a,
                % backprop y-minibatch to set delta,
                % perform descent based on calculated deltas
                n_minibatches = floor(nx/minibatch_size);
                eta_m = learning_rate/minibatch_size;
                for n = 1:n_minibatches
                    minibatch_start = (n-1)*minibatch_size + 1;
                    minibatch_end = n*minibatch_size;

                    x_minibatch = xbatch(:, minibatch_start:minibatch_end);
                    y_minibatch = ybatch(:, minibatch_start:minibatch_end);

                    obj.forward(x_minibatch);
                    obj.backprop(y_minibatch);

                    % performing gradient descent on all layers
                    for l = 2:numel(obj.layers)
                        obj.layers{l}.descend(eta_m, lambda1, lambda2);
                    end
                end % processing batch
                
                obj.total_epochs = obj.total_epochs + 1;
                progress = 100*epoch/epochs;
                delta_t = toc;
                
                if mod(progress, 1) == 0
                    msg = ['Progress: ', num2str(progress), ...
                           '%. Remaining Time: ', num2str((delta_t/epoch)*(epochs-epoch))];
                    disp(msg)
                end
                if ~isempty(obj.monitor)
                    obj.monitor.monitor(obj);
                end
                
            end %epoch-looping
        end % train
        
        
        

%% TODO clean both gradient checkers, outsource to gradientchecker class 
        function [dC_dW_backprob, dC_db_backprob, dCdW_linear, dCdB_linear] = gradient_checker(obj,x,y)
            % gradient_backprob 
            obj.layers{1}.forward(x);
            obj.layers{end}.backprop(y);
            dC_dW_backprob = []; 
            dC_db_backprob = [];
            layer = obj.layers{end};
            
            for n_run = 1:numel(x)
                while ~isempty(layer.prev) 
                    [dC_dW_back, dC_db_back] = layer.get_gradient();
                    dC_dW_back = dC_dW_back(:);
                    %[a,b] = size(dC_dW_back)% Gradient 
                    [dC_dW_backprob] = [dC_dW_backprob; dC_dW_back];
                    [dC_db_backprob] = [dC_db_backprob; dC_db_back];
                    layer = layer.prev;
                end
            end
                
            % gradient_linear
            [dCdW_linear, dCdB_linear] = obj.gradient_checking(x,y);            
        end

        function [dC_dW, dC_db] = gradient_checking(obj,x, y)
            % Global Cost Function
            cost_function = obj.layers{end}.f_cost; %(obj.layers{end}.a,y);
            %Calculates the Gradient
            dC_dW = [];
            dC_db = [];
            C_plus = 0; % Cost term for C + e 
            C_minus = 0; % Cost term for C - e
            e = 10^-4; % factor for approximation of cost
            
            for n_run = 1:numel(x) %atthe moment necessarily 
                x_run = x(n_run);
                y_run = y(n_run);
                
                for n_layer = numel(obj.layers):-1:2
                    W = obj.layers{n_layer}.W;
                    b = obj.layers{n_layer}.b;
                    % Calculation of dC_dW
                    for column_W = 1:size(W,2)
                        for row_W = 1:size(W,1)
                            %Calculate Error with positive Change
                            obj.layers{n_layer}.W(row_W, column_W) = W(row_W, column_W) + e;
                            C_plus = cost_function(obj.forward(x_run), y_run);
                            % Calculate Error with negative Change
                            obj.layers{n_layer}.W(row_W, column_W) = W(row_W, column_W) - e;
                            C_minus = cost_function(obj.forward(x_run), y_run);
                            % Estimation of Gradient:
                            dC_dW = [dC_dW, (C_plus - C_minus)/(2*e)];
                            % Reinitialize W data for next run:
                            obj.layers{n_layer}.W(row_W, column_W) = W(row_W, column_W);
                        end
                    end
                    % Calculation of dC_db
                    for column_b = 1:numel(b) 
                        obj.layers{n_layer}.b(column_b) = b(column_b) + e;
                        C_plus = cost_function(obj.forward(x_run), y_run) ;

                        obj.layers{n_layer}.b(column_b) = b(column_b) - e;
                        C_minus = cost_function(obj.forward(x_run), y_run);

                        dC_db = [dC_db, (C_plus - C_minus)/(2*e)];
                        obj.layers{n_layer}.b(column_b) = b(column_b);
                    end 
                end %n_layer
            end %n_run
        end %gradient checking
        

        function check_validity(obj, values, value_name)
            % Ckecks given values for their validity and raises apropriate
            % error
            nans = any(isnan(values));
            infs = any(isinf(values));
            outofbounds1 = any(abs(values) > 1);
            outofbounds5 = any(abs(values) > 5);
            nodouble = ~isa(values, 'double');
            
            if any([nans, infs])
                error(strcat("nan "*nans, "inf "*infs, "found within ", value_name))
            elseif outofbounds5
                error(strcat(value_name, " out of bounds (plus/minus 5). Try to normalize values between plus/minus 1"))
%             elseif outofbounds1
%                 warning(strcat(value_name, " out of bounds (plus/minus 1). Try to normalize values between plus/minus 1"))
            elseif nodouble
                error(strcat("Type Error in ", value_name, ". Expected 'double', got ", class(values), " instead"))
            end
        end
        
        
        %% Getters and Setter
        
        function p = get.hyperparams(obj)
            if isempty(obj.layers{1}.optimizer)
                p = obj.hyperparams_;
            else
                p = mergestructs(obj.hyperparams_, obj.layers{1}.optimizer.hyperparams);
            end
        end
        function C = get.cost(obj)
            C = obj.layers{end}.C;
        end

    end % methods
end % classdef
