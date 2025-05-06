clear;  close all; clc;


%% DENSO
m = sym('m', 'real');
n = sym('n', 'real');

D = m*(3*n+3+2*m*n+2*m) - (5+2*m+2*n)*(m)*(m+1)/2 + m*(m+1)*(2*m+1)/6;
expand(simplify(D));

th = sym('th');
p = sym('p');
phi = sym('phi');
N = sym('N');

D1 = subs(D, [m;n], [p*(N-phi); th+p*N]);
D1 = expand(simplify(D1));
collect(D1,N)

%% SPARSO
housegen_sp = th*(th+1+p*(1+phi))-th*(th+1)/2+(m-th)*(1+phi)*p;
prod1_sp = (m/p)*((phi+1)*p^3 - (phi+1)*p*p*(p+1)/2) + m*p^2*phi*(phi+1)/2;
prod2_sp = m*th^2+m*th+m*p*th*(1+phi)+m*(m-th)*p*(1+phi)+...
    -(m+th+1+p*(1+phi))*(th*(th+1)/2)+th*(th+1)*(2*th+1)/6+...
    +p*(1+phi)*(m-th)*(m-th+1)/2-p*(1+phi)*th*m;
collect(expand(simplify(prod2_sp)), m);

S = collect(expand(simplify(housegen_sp+prod2_sp+prod1_sp)), m);
S1 = subs(S, m, p*(N-phi));
collect(S1,N)