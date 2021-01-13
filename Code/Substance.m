% Group 3: Substance Class
% 6.10.2020 _

classdef Substance < handle
    
    properties
        name % Name of the Substance
        antoine_A % Antoine Parameter of the Substance 
        antoine_B
        antoine_C
        Pc
        Tc
        Mw
        acentric_factor
    end
    
    
    methods
        
        function obj = Substance(name)
            % CONSTRUCTOR OF CLASS SUBSTANCE
            obj.name = name;
            obj.set_parameters(); 
            obj.omega();
        end
        
        function set_parameters(obj)
            % PARAMETER DEFINITION
            % allocates the respective Parameters relevant for the 
            % Calculations based on given Substance
            
            switch obj.name
                case "methane" 
                    % Antoine Parameter for T ~ 90-190 K Prydz and Goodwin,
                    % 1972; NIST 
                    obj.antoine_A = 3.9895;
                    obj.antoine_B = 443.028;
                    obj.antoine_C = -0.49;
                    % Critical Pressure [Pa] and Temperature [K]; Nist
                    obj.Pc = 46.1e5;
                    obj.Tc = 190.6;
                    obj.Mw = 16.04e-3; % [kg/mol]
                case "ethane"
                    % Antoine Parameter for T ~ 90-145 K Carruth and
                    % Kobayashi, 1973; NIST
                    obj.antoine_A = 4.50706;
                    obj.antoine_B = 791.3;
                    obj.antoine_C = -6.422;
                    % Critical Pressure [Pa] and Temperature [K]; Nist
                    obj.Pc = 49e5;
                    obj.Tc = 305.3;
                    obj.Mw = 30.07e-3; % [kg/mol]
                case "propane"
                    % Antoine Parameter for T ~ 277.6-360.8 K Helgeson and
                    % Sage, 1967; NIST
                    obj.antoine_A = 4.53678;
                    obj.antoine_B = 1149.36;
                    obj.antoine_C = 24.906;
                    % Critical Pressure [Pa] and Temperature [K]; Nist
                    obj.Pc = 42.5e5;
                    obj.Tc = 369.9;
                    obj.Mw = 44.10e-3; % [kg/mol]
                case "butane"
                    % Antoine Parameter for T ~ 272.66-425 K Das, Reed, et
                    % al., 1973, NIST
                    obj.antoine_A = 4.35576;
                    obj.antoine_B = 1175.581;
                    obj.antoine_C = -2.071;
                    % Critical Pressure [Pa] and Temperature [K]; Nist
                    obj.Pc = 38.0e5;
                    obj.Tc = 425;
                    obj.Mw = 58.12e-3; % [kg/mol]
                case "pentane"
                    % Antoine Parameter for T ~ 268.8-341.37K Osborn and
                    % Douslin, 1974; NIST
                    obj.antoine_A = 3.9892;
                    obj.antoine_B = 1070.617;
                    obj.antoine_C = -40.454	;
                    % Critical Pressure [Pa] and Temperature [K]; Nist
                    obj.Pc = 33.6e5;
                    obj.Tc = 469.8;
                    obj.Mw = 72.15e-3; % [kg/mol]
                case "hexane"
                    % Antoine Parameter for T ~ 177.70-264.93K Carruth and
                    % Kobayashi, 1973; NIST
                    obj.antoine_A = 3.45604;
                    obj.antoine_B = 1044.038;
                    obj.antoine_C = -53.893;
                    % Critical Pressure [Pa] and Temperature [K]; Nist
                    obj.Pc = 30.2e5;
                    obj.Tc = 507.6;
                    obj.Mw = 86.18e-3; % [kg/mol]
                otherwise
                    error(['Substance ', obj.name, ' not implemented'])
            end % Switch
        end % Parameter
        
        function p = P_antoine(obj, T)
            % Calculating Vapor-Pressure [Pa] at given Temperature [K] 
            % according to Antoine Equation
            p = 1e5 * 10.^(obj.antoine_A - obj.antoine_B./(T + obj.antoine_C));
            p = reshape(p, size(T));
        end
        
        function omega(obj)
            % Calculating acentric factor by critical pressure data and
            % Antoine-pressure at reduced temperature
            ps = obj.P_antoine(obj.Tc * 0.7);
            obj.acentric_factor = -log10(ps/obj.Pc) - 1;
        end
    
    end
end