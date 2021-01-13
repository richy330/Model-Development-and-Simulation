% _
classdef PengRobinson_static < handle
  
    
    methods(Static)
                       
        function [Ts,Ps,Vms] = get_vaporpressure(T, substance)
        
        % Calculates the Saturation Pressure at a given Temperature T.
            % If no start pressure is given via the input, the pressure will
            % be set to the saturation pressure calculated with the
            % Antoine-equation at the given Temperature. Every value in
            % SI-Units.
            %
            % Input        Pressure_Calculation(T,P)
            %
            %  -T:         "vector"  (E.g a temperature span in the range of
            %                        interest. Maximal from tripple point to critical point)
            %  -p:         "vector"  (Vector of start values for the
            %                        pressure. Must be the same size as T. Possibility to dispense
            %                        given with the following if-loop.)
            %
            % Output
            %
            %  -Ts:         "vector" (Same size as T, Saturation
            %                        Temperature. I.e. the given temperature at the input) 
            %  -Ps:         "vector" (Same size as T, Calculated Saturation
            %                        Pressure for the temperatures given in the input)
            %  -Vms:        "vector" (Two colums: first one is the molar
            %                        volume of the liquid, second one is the molar volume of the
            %                        vapor. Row size indentical with T.)
            
            if ~isvector(T)
                error("T must be vector")
            end

            R = 8.31433;              % [J/mol K]
            
            size_T = size(T);
            T = T(:);
            P = substance.P_antoine(T);       % p-start according to Antoine

            Tc = substance.Tc;      % [K]
            Pc = substance.Pc;      % [Pa]
            acentric = substance.acentric_factor; 
            
            
            a_crit = 0.45724 * R^2 * Tc^2 / Pc;                         % PR Equ 9
            b_crit = 0.07780 * R * Tc / Pc;                             % PR Equ 10
            kappa = 0.37464 + 1.54226*acentric - 0.26992*acentric^2;    % PR Equ 18
                               

            alpha = @(Tr) (1 + kappa.*(1 - Tr.^0.5)).^2;                % PR Equ 17
            a = @(T) a_crit .* alpha(T./substance.Tc);                  % PR Equ 12
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
                 P = real(P.*phi_liquid./phi_vapor);
                 
                 n= n+1;
                 if n >=100, break, end
                 
            end %While-Loop
            
            Ts = T(converged);
            Ps = P(converged);
            Vms = Vm(converged,:);
            
            if size_T(1) == 1
                Ts = reshape(Ts, 1, []);
                Ps = reshape(Ps, 1, []);
                Vms = reshape(Vms, 1, []);
            end
                                    
        end %Pressure_Calculation
    end %mehtods  
end %class end