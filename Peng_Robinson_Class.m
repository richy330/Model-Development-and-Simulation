classdef Peng_Robinson_Class < handle
    
    properties
       Substance 
       Ts
       Ps
       Vms
         
    end %Properies
    
    methods
        
        function obj = Peng_Robinson_Class(Substance1)
            
            obj.Substance = Substance(Substance1);
            
        end %Constructor
        
        function Pressure_Calculation(obj,T,P)
            %Caclulates the Saturation Pressure at a given Temperature T.
            %If no start pressure is given via the input, the pressure will
            %be set to the saturation pressure calculated with the
            %Antoine-equation at the given Temperature. Every value in
            %SI-Units.
            %
            %Input        Pressure_Calculation(T,P)
            %
            % -T:         "vector"  (E.g a temperature span in the range of
            %                        interest. Maximal from tripple point to critical point)
            % -p:         "vector"  (Vector of start values for the
            %                        pressure. Must be the same size as T. Possibility to dispense
            %                        given with the following if-loop.)
            %
            %Output
            %
            % -Ts:         "vector" (Same size as T minus all not converged and complex values, Saturation
            %                        Temperature. I.e. the given temperature at the input) 
            % -Ps:         "vector" (Same size as T minus all not converged and complex values, Calculated Saturation
            %                        Pressure for the temperatures given in the input)
            % -Vms:        "vector" (Two colums: first one is the molar
            %                        volume of the liquid, second one is the molar volume of the
            %                        vapor. Row size indentical with T minus all not converged and complex values.)
            
            T = T(:);
            if nargin < 3 || isempty(P)                                 
                P = obj.Substance.P_antoine(T);       %If p is not given in the input it will be approximated here
            end
            
            P = P(:);
            Tc = obj.Substance.Tc;      % [K]
            Pc = obj.Substance.Pc;      % [Pa]
            acentric = obj.Substance.acentric_factor; 
            
            R = 8.31433;              % [J/mol K]
            
            a_crit = 0.45724 * R^2 * Tc^2 / Pc;                         % PR Equ 9
            b_crit = 0.07780 * R * Tc / Pc;                             % PR Equ 10
            kappa = 0.37464 + 1.54226*acentric - 0.26992*acentric^2;    % PR Equ 18
                               

            alpha = @(Tr) (1 + kappa.*(1 - Tr.^0.5)).^2;                % PR Equ 17
            a = @(T) a_crit .* alpha(T./obj.Substance.Tc);              % PR Equ 12
            b = b_crit;                                                 % PR Equ 13

            n = 0;
            converged = false(size(T));

            while ~all(converged)
                
                A = a(T) .* P ./ (R^2 .* T.^2);                         % PR Equ 6
                B = b * P ./ (R * T);                                   % PR Equ 7
                
                %CARDANO ALGORITHM:
                %Defining the reduced form:

                 A_card = 1;
                 B_card = -(1 - B);
                 C_card = +(A - 3*B.^2 - 2*B);
                 D_card = -(A.*B - B.^2 - B.^3);

                 a_card = B_card ./ A_card;
                 b_card = C_card ./ A_card;
                 c_card = D_card ./ A_card;
                 p_card = b_card - (a_card.^2 / 3);
                 q_card = (2*a_card.^3 / 27) - (a_card.*b_card/3) + c_card;

                 %Solving the Reduced Form
                 
                 delta_card = (q_card/2).^2 + (p_card/3).^3;
                 u = (-q_card/2 + delta_card.^0.5).^(1/3);
                 v = (-q_card/2 - delta_card.^0.5).^(1/3);
                 
                 epsilon_1 = -0.5 + 0.5i * 3^0.5;
                 epsilon_2 = epsilon_1^2;
                 
                 %Differenting betweeen the delta_card values. Namley < or
                 %< than 0. This is done with logical indexing in the
                 %following few lines
                                
                 Z = nan(size(T,1),3);
                 L_Case1 = delta_card <= 0;          %Case if p_card < 0 and delta_card <= 0:  3 real solutions
                 L_Case2 = delta_card > 0;           %CAse if p_card < 0 and delta_card > 0 :  1 real and 2 complex solutions
                 
                 
                 Z(L_Case2,1) = (u(L_Case2)+v(L_Case2))-a_card(L_Case2)/3;
                 Z(L_Case2,2) = (u(L_Case2)*epsilon_1 + v(L_Case2)*epsilon_2)-a_card(L_Case2)/3;
                 Z(L_Case2,3) = (u(L_Case2)*epsilon_2 + v(L_Case2)*epsilon_1)-a_card(L_Case2)/3;
                 
                 Z(L_Case1,1) = ( sqrt(-(4/3).*p_card(L_Case1)).*cos((1/3).*acos(-(q_card(L_Case1)./2).*sqrt(-27./p_card(L_Case1).^3))))       - a_card(L_Case1)./3; 
                 Z(L_Case1,2) = (-sqrt(-(4/3).*p_card(L_Case1)).*cos((1/3).*acos(-(q_card(L_Case1)./2).*sqrt(-27./p_card(L_Case1).^3))+pi./3)) - a_card(L_Case1)./3;
                 Z(L_Case1,3) = (-sqrt(-(4/3).*p_card(L_Case1)).*cos((1/3).*acos(-(q_card(L_Case1)./2).*sqrt(-27./p_card(L_Case1).^3))-pi./3)) - a_card(L_Case1)./3;
                                  
                 Vm = (T.*Z.*R)./P;
                 vm_liquid = min(Vm,[],2);
                 vm_vapor  = max(Vm,[],2);
                 Vm = [vm_liquid, vm_vapor];
                                                   
                 % Calculation of fugacities
                 phi = @(Z) exp(Z - 1 - log(Z - B) - A./(2*2^0.5*B).*log((Z+2.414*B)./(Z-0.414*B)));
                 
                 Z_liquid = P.*vm_liquid ./ (R*T);
                 Z_vapor = P.*vm_vapor ./ (R*T);
                 phi_liquid = phi(Z_liquid);
                 phi_vapor = phi(Z_vapor);
                 
                 converged = (abs(phi_liquid.*P - phi_vapor.*P)) <= 1;
                 P = P.*phi_liquid./phi_vapor;
                 
                 n= n+1;
                 if n >=100, break, end
                 
            end %While-Loop
                        
            Ts1 = T(converged);
            Ps1 = P(converged);
            Vms1 = Vm(converged,:);
                  
            Real_Ps = imag(Ps1)==0;                                     %Exclusion of all Parts of the results with a imaginary part.
            Real_Vms = imag(Vms1)==0;
            Real_Ts = Real_Ps;
            
            obj.Ts = Ts1(Real_Ts);
            obj.Ps = Ps1(Real_Ps);
            obj.Vms = Vms1(Real_Vms);
            
            
        end %Pressure_Calculation
        
        
        function Temperature_Calculation(obj,P,T)
            %             %Caclulates the Saturation Temperature at a given Pressure P.
            %             %If no start Temperature is given via the input, the Temperature will
            %             %be set to the saturation Temperature calculated with the
            %             %Antoine-equation at the given Pressure. Every value in
            %             %SI-Units.
            %             %
            %             %Input        Temperature_Calculation(P,T)
            %             %
            %             % -T:         "vector"  (Start Temperature, possibility of dispense given with following if loop)
            %             % -p:         "vector"  (Vector of pressures for which the corresponding
            %             %                        temperatures should be calculated. Maximal from
            %             %                        tripple point pressure to critical point pressure)
            %             %
            %             %Output
            %             %
            %             % -Ts:         "vector" (Same size as p minus non converged and complex values, Calculated Saturation Temperature for
            %             %                        the pressure given in the input)
            %             % -Ps:         "vector" (Same size as p minus non converged and complex values, Saturation
            %             %                        Pressure given in the input)
            %             % -Vms:        "vector" (Two colums: first one is the molar
            %             %                        volume of the liquid, second one is the molar volume of the
            %             %                        vapor. Row size indentical with p minus non converged and complex values.)
            
            P=P(:);
            if nargin < 3 || isempty(T)
                T = (obj.Substance.antoine_B./(obj.Substance.antoine_A-log10(P./1e+5))) - obj.Substance.antoine_C;
            end
            
            T = T(:);
            Tc = obj.Substance.Tc;      % [K]
            Pc = obj.Substance.Pc;      % [Pa]
            acentric = obj.Substance.acentric_factor;
            
            R = 8.31433;              % [J/mol K]
            
            
            a_crit = 0.45724 * R^2 * Tc^2 / Pc;                         % PR Equ 9
            b_crit = 0.07780 * R * Tc / Pc;                             % PR Equ 10
            kappa = 0.37464 + 1.54226*acentric - 0.26992*acentric^2;    % PR Equ 18
            
            n = 0;
            converged = false(size(P));
            
            while ~all(converged)
                
                alpha = @(Tr) (1 + kappa.*(1 - Tr.^0.5)).^2;                % PR Equ 17
                a = @(T) a_crit .* alpha(T./obj.Substance.Tc);              % PR Equ 12
                b = b_crit;                                                 % PR Equ 13
                
                A = a(T) .* P ./ (R^2 .* T.^2);                         % PR Equ 6
                B = b * P ./ (R * T);                                   % PR Equ 7
                
                %CARDANO ALGORITHM:
                %Defining the reduced form:
                
                A_card = 1;
                B_card = -(1 - B);
                C_card = +(A - 3*B.^2 - 2*B);
                D_card = -(A.*B - B.^2 - B.^3);
                
                a_card = B_card ./ A_card;
                b_card = C_card ./ A_card;
                c_card = D_card ./ A_card;
                p_card = b_card - (a_card.^2 / 3);
                q_card = (2*a_card.^3 / 27) - (a_card.*b_card/3) + c_card;
                
                %Solving the Reduced Form
                
                delta_card = (q_card/2).^2 + (p_card/3).^3;
                u = (-q_card/2 + delta_card.^0.5).^(1/3);
                v = (-q_card/2 - delta_card.^0.5).^(1/3);
                
                epsilon_1 = -0.5 + 0.5i * 3^0.5;
                epsilon_2 = epsilon_1^2;
                
                %Differenting betweeen the delta_card values. Namley < or
                %< than 0. This is done with logical indexing in the
                %following few lines
                
                Z = nan(size(P,1),3);
                L_Case1 = delta_card <= 0;          %Case if p_card < 0 and delta_card <= 0:  3 real solutions
                L_Case2 = delta_card > 0;           %CAse if p_card < 0 and delta_card > 0 :  1 real and 2 complex solutions
                
                
                Z(L_Case2,1) = (u(L_Case2)+v(L_Case2))-a_card(L_Case2)/3;
                Z(L_Case2,2) = (u(L_Case2)*epsilon_1 + v(L_Case2)*epsilon_2)-a_card(L_Case2)/3;
                Z(L_Case2,3) = (u(L_Case2)*epsilon_2 + v(L_Case2)*epsilon_1)-a_card(L_Case2)/3;
                
                Z(L_Case1,1) = ( sqrt(-(4/3).*p_card(L_Case1)).*cos((1/3).*acos(-(q_card(L_Case1)./2).*sqrt(-27./p_card(L_Case1).^3))))       - a_card(L_Case1)./3;
                Z(L_Case1,2) = (-sqrt(-(4/3).*p_card(L_Case1)).*cos((1/3).*acos(-(q_card(L_Case1)./2).*sqrt(-27./p_card(L_Case1).^3))+pi./3)) - a_card(L_Case1)./3;
                Z(L_Case1,3) = (-sqrt(-(4/3).*p_card(L_Case1)).*cos((1/3).*acos(-(q_card(L_Case1)./2).*sqrt(-27./p_card(L_Case1).^3))-pi./3)) - a_card(L_Case1)./3;
                
                Vm = (T.*Z.*R)./P;
                vm_liquid = min(Vm,[],2);
                vm_vapor  = max(Vm,[],2);
                Vm = [vm_liquid, vm_vapor];
                
                % Calculation of fugacities
                phi = @(Z) exp(Z - 1 - log(Z - B) - A./(2*2^0.5*B).*log((Z+2.414*B)./(Z-0.414*B)));
                
                Z_liquid = P.*vm_liquid ./ (R*T);
                Z_vapor = P.*vm_vapor ./ (R*T);
                phi_liquid = phi(Z_liquid);
                phi_vapor = phi(Z_vapor);
                
                converged = (abs(phi_liquid - phi_vapor)) <= 0.01;         %!!!!!!!!!!!!!!!!!!!!!!!!
                
                T = T.*phi_liquid./phi_vapor;                              %!!!!!!!!!!!!!!!!!!!!!!!!
                
                n= n+1;
                if n >=100; break, end
                
            end %While-Loop
            
            Ts1 = T(converged);
            Ps1 = P(converged);
            Vms1 = Vm(converged,:);
            
            Real_Ts = imag(Ts1)==0;
            Real_Ps = Real_Ts;                                     %Exclusion of all Parts of the results with a imaginary part.
            Real_Vms = imag(Vms1)==0;
            
            
            obj.Ts = Ts1(Real_Ts);
            obj.Ps = Ps1(Real_Ps);
            obj.Vms = Vms1(Real_Vms);
            
        end %Temperature_Calculation
    end %mehtods  
end %class end