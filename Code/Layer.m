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
        a {mustBeFinite, mustBeReal, mustBeNumeric}                      % activation = activ_func(z)
        W {mustBeFinite, mustBeReal, mustBeNumeric}                       % weight-matrix
        b {mustBeFinite, mustBeReal, mustBeNumeric}
        z {mustBeFinite, mustBeReal, mustBeNumeric}                       % weighted input
        dsigma_dz {mustBeFinite, mustBeReal, mustBeNumeric}               % gradient of sigma function with respect to z
        delta {mustBeFinite, mustBeReal, mustBeNumeric}                  % error-vector
        n_inputs {mustBeFinite, mustBeInteger}
        n_neurons {mustBeFinite, mustBeInteger}
        % Functions:
        activ_func
        cost_func
        optimizer
        % Linked List Attributes:
        next                    % Next Layer inside of the NN
        prev                    % Previous Layer inside of the NN
    end
    

    methods
        %% Setter functions
        function obj = set.b(obj, b)
            [m,n] = size(b);
            if m ~= obj.n_neurons
                error("Bias-vector has wrong dimension")
            end
            obj.b = b;
        end
        function obj = set.a(obj, a)
            [m,n] = size(a);
            if m ~= obj.n_neurons
                error(['a has wrong number of rows. ', 'Should be ', num2str(obj.n_neurons), ', received ', num2str(m)])
            end
            obj.a = a;
        end
        function obj = set.z(obj, z)
            [m, n] = size(z);
            if m ~= obj.n_neurons
                error(["z has wrong number of rows. "...
                    "Should be ", num2str(obj.n_neurons), ", received ", num2str(m)])
            end
            obj.z = z;
        end
        function obj = set.dsigma_dz(obj, dsigma_dz)
            [m, n] = size(dsigma_dz);
            if m ~= obj.n_neurons
                error(["dsigma_dz has wrong number of rows. "...
                    "Should be ", num2str(obj.n_neurons), ", received ", num2str(m)])
            end
            obj.dsigma_dz = dsigma_dz;
        end
        function obj = set.delta(obj, delta)
            [m,n] = size(delta);
            if m ~= obj.n_neurons
                error(["delta has wrong number of rows. "...
                    "Should be ", num2str(obj.n_neurons), ", received ", num2str(m)])
            end
            obj.delta = delta;
        end
        
        % METHODS
        %% Constructor
        function obj = Layer(n_inputs, n_neurons, activ_func, cost_func, optimizer)
            % CONSTRUCTOR OF CLASS LAYER
            if nargin < 3 || isempty(activ_func)
                activ_func = "";
            end
            obj.n_inputs = n_inputs;
            obj.n_neurons = n_neurons;
            obj.W = randn(n_inputs, n_neurons)*sqrt(1/obj.n_inputs); % CHANGE
            obj.b = zeros(n_neurons, 1);
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
                obj.delta = obj.cost_func.derivative(obj.a, y) .* obj.dsigma_dz;
            elseif ~isempty(obj.prev)
                obj.delta = (obj.next.W * obj.next.delta) .* obj.dsigma_dz;
            end
            
            if ~isempty(obj.prev)
                obj.prev.backprop()
            end
        end % backpropagation
        
        %% Get gradient
%         function [dCdW, dCdb] = get_gradient(obj)
%             dCdW = [obj.delta * obj.prev.a']';
%             dCdb = sum(obj.delta, 2);
%         end % get gradient
%             
        %% Descend
        function descend(obj, eta_m, lambda)
            obj.optimizer.descend(obj, eta_m, lambda)
%             [dCdW, dCdb] = obj.get_gradient();
%             
%             %obj.W = (1 - eta_m*lambda)*obj.W - eta_m * dCdW; %% L2 regularization
%             obj.W = obj.W - eta_m * dCdW; %% no regularization
%             obj.b = obj.b - eta_m * dCdb;
        end % gradient descent


        
%% Helper functions
%         function f_definition(obj, activ_func)
%             % SETTING ACTIVATION AND ACTIVATIONDERIVATIVE FUNCTION
%             % helper function
%             % sets obj.f_sigma and obj.f_sigmaDer
%             % according to string passed to constructor
%             % ATTENTION:
%             % when adding new functions, make sure they return column
%             % vectors!
%             switch activ_func
%                 case "ReLU" %Rectified linear unit
%                     obj.f_sigma = @(x) max(0, x);
%                     obj.f_sigmaDer = @(x) (x > 0);
%                 case "sigmoid"
%                     obj.f_sigma = @(x) 1 ./ (1+exp(-x));
%                     obj.f_sigmaDer = @(x) (exp(x) ./ (1+exp(x)).^2);    
%                 case ""
%                     obj.f_sigma = @(x) x;
%                     obj.f_sigmaDer = @(x) ones(size(x));
%             end % switch
%         end % f_definition

    end % methods
end % classdef
