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
%% MSE Based Optimal State Estimation
Q1 = 0.005;     % Process noise covariance 
Q =  Q1*eye(4);

R1 = 0.05;       % Measurement noise covariance 
R = R1 * eye(4);

ChN = 0.002;     % Channel noise covariance 

t = (0:0.1:200)';


% defining noise vakues based on covariance matrices
rng(10,'twister');
w = sqrt(Q1)*randn(length(t),1);    %Additive Process noise
v = sqrt(R1)*randn(length(t),1);    %Additive Measurement noise
ak = sqrt(ChN)*randn(length(t),1);  %Additive Channel noise

X_true = zeros(4,length(t));
X_true(:,1) = [5;-5;0.7;1];         %Initial Condition for True States

Y_true = zeros(4,length(t));
Y_true(:,1) = sys_d.C * X_true(:,1) + v(1);

X_hat = zeros(4,length(t));         
X_hat(:,1) = [7;-8;-10;15];         %Initial Condition for Estimated States

P = zeros(4,4,length(t));
P(:,:,1) = 0.02*eye(4);             % Initial Estimation error covariance

P_mad = inv(sys_d.C * P(:,:,1) * sys_d.C');

%%% Defining correction gain
K = zeros(4,4,length(t));
K(:,:,1) = P(:,:,1) * sys_d.C' * inv(sys_d.C * P(:,:,1) * sys_d.C' + R);  


Tk = randn; % Attack Value

for i=2:length(t)
    

    X_true(:,i) = sys_d.A * X_true(:,i-1) + sys_d.B * [ exp(-10*(i-1)) ; sin((i-1)/2) ] + w(i-1)*[1;1;1;1];
    Y_true(:,i) = sys_d.C * X_true(:,i) +  v(i);
    
    
    
    P(:,:,i) = sys_d.A * P(:,:,i-1) * sys_d.A' + Q + P(:,:,1) * sys_d.C' * (P_mad - Tk * P_mad - P_mad * Tk) * sys_d.C * P(:,:,1);
    K(:,:,i) = P(:,:,i) * sys_d.C' * inv(sys_d.C * P(:,:,i) * sys_d.C' + R); 
    X_hat(:,i) = sys_d.A * X_hat(:,i-1) + K(:,:,i) * ( Tk * (Y_true(:,i) - sys_d.C * sys_d.A * X_hat(:,i-1)) + ak(i));

end





figure(1)
subplot(211)
plot(1:length(t),X_true(1,:),'b',LineWidth=1)
hold on
plot(1:length(t),X_hat(1,:),'r',LineWidth=1)
legend('True state (\beta)' , 'Estimated state (\beta)')
grid on
xlabel('Number of Iteration')
subplot(212)
plot(1:length(t) , abs(X_true(1,:)-X_hat(1,:)),'black',LineWidth=1.2);
grid on
legend('Estimation Error')
xlabel('Number of Iteration')

figure(2)
subplot(211)
plot(1:length(t),X_true(2,:),'b',LineWidth=1)
hold on
plot(1:length(t),X_hat(2,:),'r',LineWidth=1)
legend('True state (\gamma)' , 'Estimated state (\gamma)')
grid on
xlabel('Number of Iteration')
subplot(212)
plot(1:length(t) , abs(X_true(2,:)-X_hat(2,:)),'black',LineWidth=1.2);
grid on
legend('Estimation Error')
xlabel('Number of Iteration')

figure(3)
subplot(211)
plot(1:length(t),X_true(3,:),'b',LineWidth=1)
hold on
plot(1:length(t),X_hat(3,:),'r',LineWidth=1)
legend('True state (\psi)' , 'Estimated state (\psi)')
grid on
xlabel('Number of Iteration')
subplot(212)
plot(1:length(t) , abs(X_true(3,:)-X_hat(3,:)),'black',LineWidth=1.2);
grid on
legend('Estimation Error')
xlabel('Number of Iteration')

figure(4)
subplot(211)
plot(1:length(t),X_true(4,:),'b',LineWidth=1)
hold on
plot(1:length(t),X_hat(4,:),'r',LineWidth=1)
legend('True state (y_{l})' , 'Estimated state (y_{l})')
grid on
xlabel('Number of Iteration')
subplot(212)
plot(1:length(t) , abs(X_true(4,:)-X_hat(4,:)),'black',LineWidth=1.2);
grid on
legend('Estimation Error')
xlabel('Number of Iteration')
