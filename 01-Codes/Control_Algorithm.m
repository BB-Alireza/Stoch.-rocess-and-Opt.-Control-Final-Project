clc; clear all; close all;
%% State space parameters
m = 380;               % vehivle mass (kg)
lr = 0.6;              % distance between the centre of gravity and the rear axle(m)
r = 0.22;              % wheel radious (m)
Cr = 6000;             % Rear tire cornering stiffness (N/rad)
Ts = 0.001;            % sampling time (sec)
lf = 0.8;              % distance between the centre of gravity and the front axle (m)
dr = 0.82;             % tread at rear axle (m)
Cf = 6000;             % Rear tire cornering stiffness (N/rad)
Q = 0.0005*eye(4);     % Process noise covariance
R = 0.05*eye(4);       % Measurement noise covariance
Vx = 25;               % Velocity (km / h)
l_pre = 1.5;           % The preview distance (m)
I = 136.08;            % Moment Inertia
%% defininge system matrices
Ac = [(-2*(Cr+Cf)/(m*Vx))   (2*(Cr*lr - Cf*lf)/(m*Vx*Vx))-1  0  0
      (2*(Cr*lr - Cf*lf)/I) (-2*(Cr*lr^2 + Cf*lf^2)/(I*Vx))  0  0
      0                     1                                0  0
      Vx                    l_pre                            Vx 0];

Bc = [ 2*Cf/(m*Vx) 2*Cf*lf/(I) 0 0
      0           1/I         0 0]';

Cc = [1 0 0 0
      0 1 0 0
      0 0 1 0
      0 0 0 1];

Dc = 0;

sys_c = ss(Ac , Bc , Cc , Dc) ;

sys_d = c2d(sys_c , Ts);
%% Control Algorithm (YALMIP)
P2 = sdpvar(4,4);    %defining pd matrix P2
S = sdpvar(2,4);
zeta = sdpvar;
%%% Constriants
tic
F = [P2>0 , P2==P2' , [-P2                      P2*sys_d.A'+S'*sys_d.B' P2 ;
                        sys_d.A*P2'+sys_d.B*S  -P2                      zeros(4,4) ;
                        P2                      zeros(4,4)             -zeta*eye(4)]<0];
sett = sdpsettings('solver','sedumi');
obj = zeta;
optimize(F,obj,sett);
PP = value(P2);
SS = value(S);

disp('The optimal State feedback gain is:')
G = SS*inv(PP);
disp(G)

disp('The Closed-Loop system eigenvalues:')
disp(eig(sys_d.A + sys_d.B * G))

zival = value(zeta);
disp('The Optimal Value of Zeta is:')
disp(zival)

disp(['The Ellapsed Time to Calculate the Optimal StateFeedback gain is:' num2str(toc)])
%% MSE Based Optimal State Estimation
Q1 = 0.0005;     % Process noise covariance 
Q = Q1 * eye(4);

R1 = 0.005;        % Measurement noise covariance 
R = R1 * eye(4);

ChN = 0.0001;   % Channel noise covariance 

t = (0:0.1:25)';

% defining noise vakues based on covariance matrices
rng(10,'twister');
w = sqrt(Q1)*randn(length(t),1);
v = sqrt(R1)*randn(length(t),1);
ak = sqrt(ChN)*randn(length(t),1);

X_true = zeros(4,length(t));
X_true(:,1) = [0.5;-0.4;0.1;-1];

Y_true = zeros(4,length(t));
Y_true(:,1) = sys_d.C*X_true(:,1)+v(1);

X_hat = zeros(4,length(t));
X_hat(:,1) = [0;0;0;0];

P = zeros(4,4,length(t));
P(:,:,1) = 0.02*eye(4);               % Initial error covariance

P_mad = inv(sys_d.C * P(:,:,1) * sys_d.C');

K = zeros(4,4,length(t));
K(:,:,1) = P(:,:,1) * sys_d.C' * inv(sys_d.C * P(:,:,1) * sys_d.C' + R);
Tk = randn;

for i=2:length(t)
    
    P(:,:,i) = sys_d.A * P(:,:,i-1) * sys_d.A' + Q + P(:,:,1) * sys_d.C' * (P_mad - Tk * P_mad - P_mad * Tk) * sys_d.C * P(:,:,1);
    K(:,:,i) = P(:,:,i) * sys_d.C' * inv(sys_d.C * P(:,:,i) * sys_d.C' + R); 
    X_hat(:,i) = sys_d.A * X_hat(:,i-1) + K(:,:,i) * ( Tk * (Y_true(:,i) - sys_d.C * sys_d.A * X_hat(:,i-1)) + ak(i));

    X_true(:,i) = sys_d.A * X_hat(:,i-1) + sys_d.B * G * X_hat(:,i-1) + w(i-1)*[1;1;1;1];
    Y_true(:,i) = sys_d.C * X_true(:,i) +  v(i);
    
    
    
    
end

%% plot Controlled states
figure(1)
plot(1:length(t),X_true(1,:),LineWidth=1)
legend('\color[rgb]{0 0.4470 0.7410}\bfControlled State (\beta)')
xlabel('\bfNumber of Iteration')
ylabel('\bfControlled States')
ylim([-0.1 0.7])
grid on

figure(2)
plot(1:length(t),X_true(2,:),'red',LineWidth=1)
legend('\color[rgb]{1 0 0}\bfControlled State (\gamma)')
xlabel('\bfNumber of Iteration')
ylabel('\bfControlled States')
ylim([-0.5 0.1])
hold on
grid on

figure(3)
plot(1:length(t),X_true(3,:),'black',LineWidth=1)
legend('\color[rgb]{0 0 0}\bfControlled State (\psi)')
xlabel('\bfNumber of Iteration')
ylabel('\bfControlled States')
ylim([-0.08 0.2])
grid on

figure(4)
plot(1:length(t),X_true(4,:),'magenta',LineWidth=1)
legend('\color[rgb]{1 0 1}\bfControlled State (y_{l})')
xlabel('\bfNumber of Iteration')
ylabel('\bfControlled States')
ylim([-1.5 0.25])
grid on


% legend('\color[rgb]{0 0.4470 0.7410}\bfControlled State (\beta)', ...
%        '\color[rgb]{0.8500 0.3250 0.0980}\bfControlled State (\gamma)', ...
%        '\color[rgb]{0.9290 0.6940 0.1250}\bfControlled State (\psi)', ...
%        '\color[rgb]{0.4940 0.1840 0.5560}\bfControlled State (y_{l})')