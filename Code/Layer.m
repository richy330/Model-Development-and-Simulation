% Group 3: Neuron-Layer Class
% 10.10.2020

classdef Layer < handle
    % Setting up a neuron-layer, given the number of incoming
    % neurons/inputs, number of neurons in the layer itself and the desired
    % activation function, cost function and optimizer.
    % Weights are initiated randomly, biases are initially 0
    
    properties
        % Variables:
        a {mustBeFinite, mustBeReal, mustBeNumeric}                        % activation = activ_func(z)
        b {mustBeFinite, mustBeReal, mustBeNumeric}
        z {mustBeFinite, mustBeReal, mustBeNumeric}                        % weighted input        
        W {mustBeFinite, mustBeReal, mustBeNumeric}                        % weight-matrix
        dsigma_dz {mustBeFinite, mustBeReal, mustBeNumeric}                % gradient of sigma function with respect to z
        delta {mustBeFinite, mustBeReal, mustBeNumeric}                    % error-vector
        n_inputs {mustBeFinite, mustBeInteger}                             % n_neurons from prev layer
        n_neurons {mustBeFinite, mustBeInteger}
        uuid
        % Functions:
        activ_func
        cost_func
        optimizer
        % Linked List Attributes:
        next                    % Next Layer inside of the NN
        prev                    % Previous Layer inside of the NN
    end
    

    methods
        %% Constructor
        function obj = Layer(n_inputs, n_neurons, activ_func, cost_func, optimizer)
            % CONSTRUCTOR OF CLASS LAYER
            obj.n_inputs = n_inputs;
            obj.n_neurons = n_neurons;
            obj.W = randn(n_inputs, n_neurons)*sqrt(1/obj.n_inputs);
            obj.b = zeros(n_neurons, 1);
            obj.uuid = randi([2^51, 2^52-1]);
            obj.activ_func = activ_func;
            obj.optimizer = optimizer;
            obj.cost_func = cost_func;
        end
        
        %% Forward function
        function y = forward(obj, inputs)
            % FEEDFORWARD FOR NN 
            % Calculating activation, weighted inputs and derivations
            % according to current weights and biases, given activation
            % of previous layer. First layer will have activation set to
            % user input.
            % Storing values in respective layer-parameters.
            
            if any(isnan(inputs))
                error("Input passed to forward method contains nan")
            end
            
            if isempty(obj.prev) % First Layer has activation set to user-input
                obj.a = inputs;
                obj.next.forward(obj.a);
            else  
                obj.z = obj.W'*inputs + obj.b;
                obj.a = obj.activ_func.a(obj.z);
                obj.dsigma_dz = obj.activ_func.dsigma_dz(obj.z);
                if ~isempty(obj.next) % forward whenever there are next layers
                    obj.next.forward(obj.a);
                else
                    y = obj.a;
                    return % if last Layer - Stop forward
                end
            end
        end % forward
        
        %% Backprop
        function backprop(obj, y)
            % Applying backpropagation algorithm, determining errors for
            % corresponding layer, storing error parameter 'delta'
            
            if isempty(obj.next)
                obj.delta = obj.cost_func.derivative(obj.a, y) .* obj.dsigma_dz;
            elseif ~isempty(obj.prev)
                obj.delta = (obj.next.W * obj.next.delta) .* obj.dsigma_dz;
            end
            
            if ~isempty(obj.prev)
                obj.prev.backprop()
            end
        end % backpropagation
            
        %% Descend
        function descend(obj, eta_m, lambda)
            % passing obj-reference to optimizer, who will perform descend
            % on obj
            obj.optimizer.descend(obj, eta_m, lambda)
        end
        
        
        
        %% Setter functions
        function set.b(obj, b)
            [m,n] = size(b);
            if m ~= obj.n_neurons
                error(['b has wrong number of rows. Should be ', num2str(obj.n_neurons), ', received ', num2str(m)])
            end
            obj.b = b;
        end
        function set.a(obj, a)
            [m,n] = size(a);
            if m ~= obj.n_neurons
                error(['a has wrong number of rows. Should be ', num2str(obj.n_neurons), ', received ', num2str(m)])
            end
            obj.a = a;
        end
        function set.z(obj, z)
            [m, n] = size(z);
            if m ~= obj.n_neurons
                error(["z has wrong number of rows. Should be ", num2str(obj.n_neurons), ", received ", num2str(m)])
            end
            obj.z = z;
        end
        function set.dsigma_dz(obj, dsigma_dz)
            [m, n] = size(dsigma_dz);
            if m ~= obj.n_neurons
                error(["dsigma_dz has wrong number of rows. Should be ", num2str(obj.n_neurons), ", received ", num2str(m)])
            end
            obj.dsigma_dz = dsigma_dz;
        end
        function set.delta(obj, delta)
            [m, n] = size(delta);
            if m ~= obj.n_neurons
                error(["delta has wrong number of rows. Should be ", num2str(obj.n_neurons), ", received ", num2str(m)])
            end
            obj.delta = delta;
        end
    end % methods
end % classdef
