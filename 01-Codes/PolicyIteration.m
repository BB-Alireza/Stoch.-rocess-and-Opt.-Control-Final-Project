clc; clear all; close all;
%% drfining system and cost function weight matrices
A = [0.8 1;1.1 2];
B = [0.2;1.4];
C = [0.7 0;-1 -0.5];
D = [-1;0.8];

Q = eye(2);     % states weight matrix for each step
R= 1;           % Input  weight matrix for each step
 
n = size(A , 1);
%% Implementation Policy Iteration Algorithm
nP = 100 ; % Total Iterations
L = zeros(nP , 2); L(1 , :) = [-1.4 -2.1]; % State feedback gain Initialization
P = cell(nP , 1) ; P{1} = zeros(n);        % Kernel matrix Initialization
gamma1 = 0.7;                              % Discount Factor

options = optimoptions('fmincon' , 'Display' , 'off') ; % fmincon options setting
tic
for j = 1:nP
    
    cost = @(P) PI(P , A , B , C , D , L(j , :) , Q , R , gamma1) ; % Policy Evaluation
    [Ps , Fval] = fmincon(cost , P{j}(:), [],[],[],[],[],[],[],options) ;
    
    P{j+1} = reshape(Ps , size(A)) ; 
    
    L(j+1 , :) = (R + B'*P{j}*B)^(-1)*(B'*P{j}*A); % Policy Improvement
    
    disp(['Iteration(' num2str(j) ')']);
    
    if norm(L(j+1 , :)-L(j , :)) < 1e-8  % Convergence check
       break; 
    end
end

for v=j+2:nP
    L(v,:) = L(j+1,:);
end
disp(['Elapsed Time = ' ,num2str(toc),' Seconds']);

disp('The Policy Iteration P Matrix is:')
disp(P{j+1})

disp('The Policy Iteration Method Gain is:')
K_EST = -inv(R + gamma1 * B' * P{j+1} * B + gamma1 * D' * P{j+1} * D) * (gamma1 * B' * P{j+1} * A + gamma1 * D' * P{j+1} *C);
disp(K_EST)

figure(1)
plot(1:nP,L(1:nP,1),LineWidth=1.25)
hold on
plot(1:nP,L(1:nP,2),LineWidth=1.25)
grid on
xlabel('Number of Iterations')
title('StsteFeedback gain Convergence Process')

%% simulation
X = zeros(2,nP);X(:,1)=[-2.5;5.7];
Q1 = 0.005;
w = sqrt(Q1)*randn(nP,1);    %Additive Process noise
for q=1:nP-1
    X(:,q+1) = (A - B*L(j+1,:)) * X(:,q) + (C - D*L(j+1,:))* X(:,q) * normrnd(0,0.1) + w(q);
end
figure(2)
plot(1:nP,X(1,:),'b',LineWidth=1.2)
hold on
plot(1:nP,X(2,:),'r',LineWidth=1.2)
legend('X_{1}','X_{2}')
grid on
xlabel('\bfNumber of Iteration')
ylabel('\bfSystem States')
ylim([-3 6])
%% functions

% fmincon f(x: n*1)
function z = PI(P , A , B , C , D , K , Q , R , gamma)
    
    P = reshape(P , size(A)) ; 
    M =  gamma*(A-B*K)' * P * (A-B*K) - P + Q + K'*R*K  + gamma*(C-D*K)' * P * (C-D*K) ;
    
    z = norm(M); %sum(abs(M(:))) ; %norm(M) ;
end
