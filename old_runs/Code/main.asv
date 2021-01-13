% Model Simulation: Group 3
% Main File


%% Names of previously Run Functions
Name_1 = "NN-run_{100}-design-1-10-10-1-stepsize-01";
Name_2 = "NN-run_{100}-design-1-10-10-1-stepsize-01_rerun1";
Name_3 = "NN-run_{100}-design-1-10-10-1-stepsize-01_rerun3";
Name_4 = "NN-run_{100}-design-1-10-10-1-stepsize-01_rerun4";
Name_5 = "NN-run4-design-1-10-10-1-stepsize-1_optimization100";
Name_6 = "NN-run4-design-1-10-10-1-stepsize-1_optimization150";
Name_7 = "NN-design_1_10_1";
Name_8 = "NN-design-1-40-1-reruns";
Name_9 = "NN_opt";
Name_10 = "NN_tryout";
Name1 = "NN-from_Scratch_1";

Name3 = "NN-from_Scratch_3";

%% Loading of Data to run
x = readtable("Peng_Robinson.txt");
T = (x{:,1}');
T_max = max(T);
L_1 = T > 60;
T_extra = T(L_1);
T = [T, T_extra, T_extra, T_extra, T_extra];
T = T/max(T);
P = x{:,2}';
P_max = max(P);
P_extra = P(L_1);
P = [P,P_extra, P_extra ,P_extra, P_extra];
P = P/max(P);
T = T(P>0.01);
P = P(P>0.01);
T = T(1:100:end);
P = P(1:100:end);

%% Setting up the NN to train
nn = Network([1,150,150,1],"sigmoid", "cross-entropy");
%load(Name1)
%stepsize = 1;
% average_error_prev = 10;
disp("----------------NEW RUN----------------")

%% Chance of which parameters to use
lambda = 1;
stepsize = 1;
limit = 15;
counter = inf;
average_error_prev = 0;
factor = 0.95;

tic
for i = 1:3000
   nn.train(T, P, 32, stepsize);
   average_error_new = mean(abs(P - nn.forward(T)));
   
   if mod(i,100) == 0
      disp(['RUN: ' num2str(i) ' STEPSIZE: ' num2str(stepsize) ' ERROR: ' num2str(average_error_new) ])
   end

   if average_error_prev < average_error_new
       if counter == inf
           counter = i;
           average_error_prev = average_error_new;
           %disp(['average error prev: ' num2str(average_error_prev)]) 
       elseif (i - counter) > limit
           stepsize = stepsize*factor;
           counter = inf;
           average_error_prev = average_error_new;
           %disp(['Average Error is ', num2str(average_error_new), ' old error is ' num2str(average_error_prev)])
           %disp(['New stepsize is ', num2str(stepsize)])
       end
   else
       average_error_prev = average_error_new;
   end
end
toc
counter = inf;
average_error_prev = 0;

% Name1 = "NN-from_Scratch_1";
% save(Name1, "nn");


% for i2 = 1:100
%    nn.train(T, P, 32, stepsize, lambda);
%    average_error_new = mean(abs(P - nn.forward(T)));
%    
%    if mod(i2,100) == 0
%        disp(['This is the ' num2str(i2) ' run for lambda!' ])
%        disp(['The Error is ' num2str(average_error_new)])
%    end
%    
%    if average_error_prev < average_error_new
%        if counter == inf
%            counter = i2;
%            average_error_prev = average_error_new;
%            %disp(['average error prev: ' num2str(average_error_prev)])
%        elseif (i2 - counter) > limit
%            lambda = lambda*1.1;
%            counter = inf;
%            average_error_prev = average_error_new;
%            %disp(['Average Error is ', num2str(average_error_new), ' old
%            %error is ' num2str(average_error_prev)])
%            disp(['New lambda is ', num2str(lambda)])
%        end
%    else
%        average_error_prev = average_error_new;
%    end
% end




%% Change of    
% difference = abs(P - nn.forward(T));
% average_error_new = mean(P_max*abs(P - nn.forward(T)));
% disp(["The average Error is ", num2str(average_error_new)])
% disp(["The maximum error is ", num2str(max(difference)*P_max)])
% disp(["The forward method(1) results in" num2str(nn.forward(1))])
% disp(["The forward method(0) results in" num2str(nn.forward(0))])
% %average_error_old = 0;
% if average_error_new < average_error_old
%     disp("NETWORK IMPROVED :D")
% end

% average_error_old = average_error_new;
Name3 = "NN-from_Scratch_3";
save(Name3, "nn");

% Runs that were completed


Graphical_Comparison({Name_1, Name_8, Name1, Name2}, x)