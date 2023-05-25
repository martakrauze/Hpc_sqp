clear;
clc;

load results.m
T=1;
global n=length(x)/9;
global h=T/(n-1);
t=[0:h:T];


global l1=1;
global l2=1;
global l3=1;

N=3;

l=[l1,l2,l3];
#l=[l1,l2];

theta=[]; thetadot=[]; u=[];

for i=[1:N]
    b = x([(i-1)*n+1:i*n]);
    theta=[theta,b];
    b = x([n*N+(i-1)*n+1:n*N+i*n]);
    thetadot=[thetadot,b];
    b = x([2*n*N+(i-1)*n+1:2*n*N+i*n]);
    u=[u,b];
endfor

fi=[zeros(n,N)];
for part=[1:N]
    for i=[1:n]
        for j=[1:part]
            fi(i,part)=fi(i,part)+theta(i,j);
        endfor
    endfor
endfor

trajx = [zeros(1,n)];
trajy = [zeros(1,n)];
for i=[1:N]
    trajx = [trajx; trajx(i,:) + (l(i) * cos(fi(:,i)))'];
    trajy = [trajy; trajy(i,:) + (l(i) * sin(fi(:,i)))'];
endfor

xkon=l*cos(fi)';
ykon=l*sin(fi)';

function plotangle(t,theta)
plot(t,theta,"LineWidth", 2);
title("kąt")
xlabel("t[s]")
ylabel('\theta [rad]', 'interpreter', 'tex')
legend("człon1","człon2", "człon3")
set(gca, "fontsize", 17)
grid
endfunction

function plotvel(t,thetadot)
#subplot(3,2,3)
plot(t,thetadot,"LineWidth", 2);
title("prędkość kątowa")
xlabel("t[s]")
ylabel('\omega [rad/s]', 'interpreter', 'tex')
legend("człon1","człon2", "człon3")
set(gca, "fontsize", 17)
grid
endfunction

function plotmom(t,u)
#subplot(3,2,5)
plot(t,u,"LineWidth", 2);
title("moment zadany")
xlabel("t[s]")
ylabel("M[Nm]")
legend("człon1","człon2", "człon3")
set(gca, "fontsize", 17)
grid
endfunction

function plottraj(trajx,trajy,xkon,ykon)
global n
#subplot(3,2,2:2:6)
plot(trajx(:,1),trajy(:,1),"LineWidth", 4,trajx(:,n),trajy(:,n),"black","LineWidth", 4)
#plot(trajx,trajy,"black","LineWidth", 4)
hold;
plot(xkon,ykon,"LineWidth", 2)
title("trajektoria")
xlabel("x[m]")
ylabel("y[m]")
axis("equal")
set(gca, "fontsize", 17)
grid
endfunction

subplot(3,1,1)
plotangle(t,theta)
subplot(3,1,2)
plotvel(t,thetadot)
subplot(3,1,3)
plotmom(t,u)