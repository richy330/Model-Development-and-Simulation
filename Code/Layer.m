% Group 3: Neuron-Layer Class
% 10.10.2020

classdef Layer < handle
    % Setting up a neuron-layer, given the number of incoming
    % neurons/inputs, number of neurons in the layer itself and the desired
    % activation function, passed by string. If no/empty string or empty
    % argument is given, output will be set unchanged -> a = f(z) = z.
    % Weights are initiated randomly, biases are initially 0
    
    properties
        % PROPERTIES OF CLASS LAYER
        % Variables:
        z                       % weighted input
        dsigma_dz               % gradient of sigma function with respect to z
        a                       % activation = activ_func(z)
        W                       % weight-matrix
        b                       % bias-vector
        delta                   % error-vector
        n_inputs                % number of inputs
        n_neurons               % number of neurons
        % Functions:
        f_sigma                 % Activation Function, sigma(z)
        f_sigmaDer              % Derivation Function(sigma)/dz
        f_cost
        f_costDer
        % Linked List Attributes:
        next                    % Next Layer inside of the NN
        prev                    % Previous Layer inside of the NN
    end
    

    methods
        %% Setter functions
        function obj = set.W(obj, W)
            if any(isnan(W))
                error("Weight set to nan")
            elseif any(isinf(W))
                error("Weight set to inf")
            elseif isnumeric(W)
                obj.W = W;
            else
                error("Weight must be numerical")
            end
        end
        function obj = set.b(obj, b)
            [m,n] = size(b);
            if any(isnan(b))
                error("Bias set to nan")
            elseif any(isinf(b))
                error("Bias set to inf")
            elseif m ~= obj.n_neurons
                error("Bias-vector has wrong dimension")
            else
                obj.b = b;
            end
        end
        function obj = set.a(obj, a)
            [m,n] = size(a);
            if any(isnan(a))
                error("Activation set to nan")
            elseif any(isinf(a))
                error("Activation set to inf")
            elseif m ~= obj.n_neurons
                error("Activation-vector has wrong dimension")
            else
                obj.a = a;
            end
        end
        function obj = set.z(obj, z)
            [m, n] = size(z);
            if any(isnan(z))
                error("Weighted input set to nan")
            elseif any(isinf(z))
                error("Weighted input set to inf")
            elseif m ~= obj.n_neurons
                error(strcat("z has wrong number of rows. Should be [", num2str(obj.n_neurons), ", x], received [", num2str(size(z)), "]"))
            else
                obj.z = z;
            end
        end
        function obj = set.dsigma_dz(obj, dsigma_dz)
            [m, n] = size(dsigma_dz);
            if any(isnan(dsigma_dz))
                error("dsigma_dz set to nan")
            elseif any(isinf(dsigma_dz))
                error("dsigma_dz set to inf")
            elseif m ~= obj.n_neurons
                error(strcat("dsigma_dz has wrong number of rows. Should be [", num2str(obj.n_neurons), ", x], received [", num2str(size(dsigma_dz)), "]"))
            elseif isempty(dsigma_dz)
                error("dsigma_dz set to empty value")
            else
                obj.dsigma_dz = dsigma_dz;
            end
        end
        function obj = set.delta(obj, delta)
            [m,n] = size(delta);
            if any(isnan(delta))
                error("Delta set to nan")
            elseif any(isinf(delta))
                error("Delta set to inf")
            elseif m ~= obj.n_neurons
                error("Delta-vector has wrong dimension")
            else
                obj.delta = delta;
            end
        end
        
        % METHODS
        %% Constructor
        function obj = Layer(n_inputs, n_neurons, activ_func)
            % CONSTRUCTOR OF CLASS LAYER
            if nargin < 3 || isempty(activ_func)
                activ_func = "";
            end
            obj.n_inputs = n_inputs;
            obj.n_neurons = n_neurons;
            obj.W = randn(n_inputs, n_neurons); % CHANGE
            obj.b = zeros(n_neurons, 1);
            obj.f_definition(activ_func); % Defines Activitation Function
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
                % obj.dsigma_dz = obj.f_sigmaDer(obj.z);
                obj.next.forward(obj.a); % Forwarding activation to next Layers
            else  
                obj.z = obj.W'*inputs + obj.b;
                obj.a = obj.f_sigma(obj.z);
                obj.dsigma_dz = obj.f_sigmaDer(obj.z);
                if ~isempty(obj.next) % forward whenever there are next layers
                    obj.next.forward(obj.a);
                else
                    y = obj.a; % NEW TEST!
                    return % if last Layer - Stop forward
                end
            end
        end % forward
        
        %% Backprob
        function backprop(obj, y)
            % Applying backpropagation algorithm, determining errors for
            % corresponding layer, storing error parameter 'delta'
            
            if isempty(obj.next)
                obj.delta = obj.f_costDer(obj.a, y) .* obj.dsigma_dz;
            elseif ~isempty(obj.prev)
                obj.delta = (obj.next.W * obj.next.delta) .* obj.dsigma_dz;
            end
            
            if ~isempty(obj.prev)
                obj.prev.backprop()
            end
        end % backpropagation
        
        %% Get gradient
        function [dCdW, dCdb] = get_gradient(obj)
            dCdW = [obj.delta * obj.prev.a']';
            dCdb = sum(obj.delta, 2);
        end % get gradient
            
        %% Descend
        function descend(obj, eta_m, lambda)
            [dCdW, dCdb] = obj.get_gradient();
            obj.W = (1 - eta_m*lambda)*obj.W - eta_m * dCdW; %% L2 regularization
            % obj.W = obj.W - eta_m * dCdW; %% no regularization
            obj.b = obj.b - eta_m * dCdb;
        end % gradient descent


        
%% Helper functions
        function f_definition(obj, activ_func)
            % SETTING ACTIVATION AND ACTIVATIONDERIVATIVE FUNCTION
            % helper function
            % sets obj.f_sigma and obj.f_sigmaDer
            % according to string passed to constructor
            % ATTENTION:
            % when adding new functions, make sure they return column
            % vectors!
            switch activ_func
                case "ReLU" %Rectified linear unit
                    obj.f_sigma = @(x) max(0, x);
                    obj.f_sigmaDer = @(x) (x > 0);
                case "sigmoid"
                    obj.f_sigma = @(x) 1 ./ (1+exp(-x));
                    obj.f_sigmaDer = @(x) (exp(x) ./ (1+exp(x)).^2);    
                case ""
                    obj.f_sigma = @(x) x;
                    obj.f_sigmaDer = @(x) ones(size(x));
            end % switch
        end % f_definition

    end % methods
end % classdef
