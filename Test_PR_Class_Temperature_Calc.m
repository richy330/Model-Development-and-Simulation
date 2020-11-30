 clear all
 close all
 

methane = Substance("methane");

%P = linspace(100000,4600000,3);
P=98649;

%%%%------------------------------------
           P=P(:);
           % if nargin < 3 || isempty(P)                                 
                T = (methane.antoine_B./(methane.antoine_A-log10(P/1e+5))) - methane.antoine_C;    
            %end
            
            T = T(:);
            Tc = methane.Tc;      % [K]
            Pc = methane.Pc;      % [Pa]
            acentric = methane.acentric_factor; 
            
            R = 8.31433;              % [J/mol K]
            
            
            a_crit = 0.45724 * R^2 * Tc^2 / Pc;                         % PR Equ 9
            b_crit = 0.07780 * R * Tc / Pc;                             % PR Equ 10
            kappa = 0.37464 + 1.54226*acentric - 0.26992*acentric^2;    % PR Equ 18
                         
            n = 0;
            converged = false(size(P));

           while ~all(converged)
               
               alpha = @(Tr) (1 + kappa.*(1 - Tr.^0.5)).^2;                % PR Equ 17
               a = @(T) a_crit .* alpha(T./methane.Tc);              % PR Equ 12
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
                
                 figure(1)
                 hold on
                 plot(n,T(1),'r*')
                 
                 n= n+1;
                 if n >=100; break, end
                 
            end %While-Loop
            
                hold off   
                
            Ts1 = T(converged);
            Ps1 = P(converged);
            Vms1 = Vm(converged,:);
              
            Real_Ts = imag(Ts1)==0;
            Real_Ps = Real_Ts;                                     %Exclusion of all Parts of the results with a imaginary part.
            Real_Vms = imag(Vms1)==0;
            
            
            Ts = Ts1(Real_Ts);
            Ps = Ps1(Real_Ps);
            Vms = Vms1(Real_Vms);