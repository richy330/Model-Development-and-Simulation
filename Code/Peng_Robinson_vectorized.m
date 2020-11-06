function [Ts, Ps, Vms] = Peng_Robinson_vectorized(substance, T, P_start)
% VAPOUR PRESSURE CALCULATION VIA PENG ROBINSON EOS
% Calculates the Vapor Pressure as well as molar volume of liquid and vapor 
% Temperature values exceeding critical Temperature are discarded.
% Return vectors of accepted Temperature, Saturation Pressures and molar
% Volumes.
% The Equations used are from "A New Two-Constant Equation of State", 
% Ding-Yu Peng and Donald B. Robinson


T = T(:);
if nargin < 3 || isempty(P_start) % number of arguments going in
    P_start = substance.P_antoine(T);
end

p = P_start;
R = 8.314;              % [J/mol K]
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
    A = a(T) .* p ./ (R^2 .* T.^2);                         % PR Equ 6
    B = b * p ./ (R * T);                                   % PR Equ 7

    %% Cardano - reduced form
    A_card = 1;
    B_card = -(1 - B);
    C_card = +(A - 3*B.^2 - 2*B);
    D_card = -(A.*B - B.^2 - B.^3);

    a_card = B_card ./ A_card;
    b_card = C_card ./ A_card;
    c_card = D_card ./ A_card;
    p_card = b_card - (a_card.^2 / 3);
    q_card = (2*a_card.^3 / 27) - (a_card.*b_card/3) + c_card;

    %% Cardano solution for Z
    delta_card = (q_card/2).^2 + (p_card/3).^3;
    u = (-q_card/2 + delta_card.^0.5).^(1/3);
    v = (-q_card/2 - delta_card.^0.5).^(1/3);

    epsilon_1 = -0.5 + 0.5i * 3^0.5;
    epsilon_2 = epsilon_1^2;

    z1 = u + v;
    z2 = u*epsilon_1 + v*epsilon_2;
    z3 = u*epsilon_2 + v*epsilon_1;

    Z1 = z1 - a_card/3;
    Z2 = z2 - a_card/3;
    Z3 = z3 - a_card/3;


    Vm = [Z1*R.*T ./ p, Z2*R.*T ./ p, Z3*R.*T ./ p];
    vm_liquid = min(Vm,[],2);
    vm_vapor = max(Vm,[],2);
    Vm = [vm_liquid, vm_vapor];

    %% Calculation of fugacities
    phi = @(Z) exp(Z - 1 - log(Z - B) - A./(2*2^0.5*B).*log((Z+2.414*B)./(Z-0.414*B))); % PR Equ 15

    Z_liquid = p.*vm_liquid ./ (R*T);
    Z_vapor = p.*vm_vapor ./ (R*T);
    phi_liquid = phi(Z_liquid);
    phi_vapor = phi(Z_vapor);
    
    converged = (abs(phi_liquid.*p - phi_vapor.*p)) <= 1;                               % PR Equ 16
    p = real(p.*phi_liquid./phi_vapor);
    n = n+1;
    if n >= 100
        break
    end
end
Ts = T(converged);
Ps = p(converged);
Vms = Vm(converged,:);
end

