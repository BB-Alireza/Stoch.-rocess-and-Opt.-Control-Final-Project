clc;
clear;
close all;
clc; clear all; close all;
%% drfining system and cost function weight matrices
A = [0.8 1;1.1 2];
B = [0.2;1.4];
C = [0.7 0;-1 -0.5];
D = [-1;0.8];

Q = eye(2);     % states weight matrix for each step
R= 1;           % Input  weight matrix for each step
 
n = size(A , 1);

R = 1 ;
Q = eye(n);

%% policy iteration

nP = 100 ;
L = zeros(nP , n); L(1 , :) = place(A , B , [-1.4 -2.1]);
H = cell(nP , 1) ; H{1} = zeros(n+1);
M = 20 ;
tic
for j = 1:nP
    
    PHI = [] ; 
    SAI = [] ;
    
    for  k = 1:M
            xk = randn(n,1) ;
            uk = -L(j , :)*xk + 0.01*randn;
            xk1 = A*xk+B*uk ;
        
            uk1 = -L(j , :)*xk1 ;
        
            PHI = [PHI ; ComputeZbar([xk;uk])-ComputeZbar([xk1;uk1])]; %#ok
            SAI = [SAI ; xk'*Q*xk+uk'*R*uk];%#ok
    end

    Hbar = PHI\SAI ;  %%% Least Square Solving
    
    H{j+1} = ConvertHbarToH(Hbar) ;
    
    Hxx = H{j+1}(1:n , 1:n) ; 
    Hxu = H{j+1}(1:n , n+1) ; 
    Hux = H{j+1}(1+n , 1:n) ; 
    Huu = H{j+1}(1+n , 1+n) ; 
    
    %%% Pol
    L(j+1 , :) = inv(Huu)*Hux;
    
    disp(['Iteration(' num2str(j) ')']);
    
    if norm(L(j+1 , :)-L(j , :)) < 1e-6
       break; 
    end
end

for v=j+2:nP
    L(v,:) = L(j+1,:);
end

disp(['Elapsed Time = ' ,num2str(toc),' Seconds']);



disp('Optimal Control Policy obtained by Qlearning = ');
disp(-L(j+1 , :))

%% simulation
X = zeros(2,nP);X(:,1)=[-2.5;5.7];
Q1 = 0.005;
w = sqrt(Q1)*randn(nP,1);    %Additive Process noise
for q=1:nP-1
    X(:,q+1) = (A - B*L(j+1,:)) * X(:,q) + (C - D*L(j+1,:))* X(:,q) * normrnd(0,1.2) + w(q);
end
figure(2)
plot(1:nP,X(1,:),'b',LineWidth=1.2)
hold on
plot(1:nP,X(2,:),'r',LineWidth=1.2)
legend('X_{1}','X_{2}')
grid on
xlabel('\bfNumber of Iteration')
ylabel('\bfSystem States')
ylim([-10 15])
%% plot results

Fig = figure(1) ;
Fig.Color = [0.9 0.9 0.9];


plot(1:nP , L(: , 1) , 'linewidth' , 1.2) ;
hold on
plot(1:nP , L(: , 2) , 'linewidth' , 1.2) ;
grid on
xlabel('\bfNumber of Iterations' , 'fontSize' , 12);
legend('K_{1}' , 'K_{2}')


%% functions

function Zbar = ComputeZbar(Z)  % Z =[X;U]
    Z = Z(:)'; 
    Zbar = [] ; 
    
    for i = 1:numel(Z)
        Zbar = [Zbar Z(i)*Z(i:end)]; %# ok
    end
end

function H = ConvertHbarToH(Hbar)

    H = [Hbar(1)   Hbar(2)/2    Hbar(3)/2   
         Hbar(2)/2 Hbar(4)      Hbar(5)/2   
         Hbar(3)/2 Hbar(5)/2    Hbar(6)  ];
end

