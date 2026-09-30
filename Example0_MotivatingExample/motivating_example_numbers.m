%% MOTIVATING_EXAMPLE_NUMBERS
%  Reproduces every figure quoted in the paragraph of Section 3.3 of the paper
%  ("A motivating example"), answering Reviewer 3, Comment 5.
%
%  Model (scalar, first order, n = p = q = 1):
%        y_t = th1*y_{t-1} + th2*u_{t-1},        y_1 = 0.
%
%  PART 1 -- gradient magnitudes.
%     Unconstrained (Problem (5)): d y_t/d th2 = sum_k th1^(k-1) u_{t-k}, so the
%     lag-k term carries the weight th1^(k-1).  Its dynamic range over the record
%     is |th1|^(N-2): it collapses for |th1|<1 and blows up for |th1|>1, at a
%     rate that grows with N.
%     Constrained (Problem (11)): grad_xi h_t has the four nonzeros
%     [-y_t, -u_t, -th1, 1]; the coefficients coupling the unknowns across time
%     are {1,-th1} at every t, so their spread is |th1|, independent of N.
%
%  PART 2 -- iteration counts.
%     Plain gradient descent on (5) vs. the FL-CMO iteration (16), both started
%     from theta = [0.1 0.1]' and both run at the largest stable step, counting
%     iterations to come within 0.1% of the optimal cost.
%
%  NOTE ON A SIGN.  The multiplier below is computed as
%         lambda = (J*J')\(K*h - J*gradL)
%  which is the law that yields the closed loop  dh/dt = -K h  for the plant
%  xi_dot = -gradL - grad_xi(h)*lambda.  See the note sent with this script:
%  Eq. (10) of the current manuscript carries the opposite sign on J*gradL and
%  does NOT drive the residual to zero (verified numerically).
%
%  Usage:  >> motivating_example_numbers
% -------------------------------------------------------------------------
clear; clc;  rng(0);

%% ------------------------------- settings --------------------------------
N    = 500;                  % record length                    -> paper: N
th2  = 1.0;                  % input gain                       -> paper: theta_2
TH1  = [0.90, 1.05];         % contractive / expansive          -> paper: theta_1
th1s = 0.90;                 % true theta_1 used for Part 2
nsr  = 0.01;                 % output noise, 1% of output std
th0  = [0.1; 0.1];           % initial guess for both optimizers
K    = 1.0;                  % FL-CMO gain

u    = randn(N,1);
simy = @(a,b) filter([0 b],[1 -a],u);      % y_t = a*y_{t-1} + b*u_{t-1}, y_1 = 0

outfile = fullfile(fileparts(mfilename('fullpath')),'motivating_numbers.txt');
fid = fopen(outfile,'w');
prt = @(varargin) cellfun(@(f) fprintf(f,varargin{:}), {1,fid});

prt('MOTIVATING EXAMPLE -- numbers for Section 3.3\n');
prt('N = %d, theta_2 = %g, u ~ white noise of unit variance\n\n', N, th2);

%% ============================== PART 1 ===================================
for th1 = TH1
    if abs(th1) < 1, tag = 'CONTRACTIVE'; else, tag = 'EXPANSIVE'; end

    w      = th1.^((1:N-1).'-1);                       % lag weights
    range  = log10(abs(w(end)/w(1)));                  % orders of magnitude
    nrepr  = sum(abs(w) >= eps*max(abs(w)));           % terms above resolution

    y  = simy(th1,th2);
    g1 = filter([0 1],[1 -th1], y);                    % d y_t / d th1
    g2 = filter([0 1],[1 -th1], u);                    % d y_t / d th2
    ng = hypot(g1,g2);

    prt('--- %s case: theta_1 = %g ---\n', tag, th1);
    prt('  lag weights th1^(k-1), dynamic range   : 10^(%+.1f)\n', range);
    prt('  lag terms above double precision       : %d of %d\n', nrepr, N-1);
    prt('  ||grad_theta y_t||_2 at t = 2          : %.2e\n', ng(2));
    prt('  ||grad_theta y_t||_2 at t = N          : %.2e\n', ng(N));
    prt('  constraint gradient: nonzeros per column %d, time couplings {1,-th1},\n', 4);
    prt('                       spread %.2f (independent of N)\n\n', max(1,abs(th1))/min(1,abs(th1)));
end

%% ============================== PART 2 ===================================
ytrue = simy(th1s,th2);
yt    = ytrue + nsr*std(ytrue)*randn(N,1);
cost  = @(y) sum((y-yt).^2);

opt   = optimset('TolX',1e-12,'TolFun',1e-14,'MaxFunEvals',4e4,'MaxIter',4e4);
thopt = fminsearch(@(t) cost(simy(t(1),t(2))), th0, opt);
Cstar = cost(simy(thopt(1),thopt(2)));
tgt   = 1.001*Cstar;
prt('--- optimizers: true theta = [%g %g], %g%% noise ---\n', th1s, th2, 100*nsr);
prt('  optimal cost C* = %.4f, target = %.4f (0.1%% above C*)\n', Cstar, tgt);

% ---- gradient descent on the unconstrained SEM ---------------------------
best_gd = [NaN NaN];
for al = [1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 8e-6]
    th = th0; it = NaN;
    for k = 1:4e5
        y  = simy(th(1),th(2));
        if cost(y) <= tgt, it = k; break; end
        g1 = filter([0 1],[1 -th(1)], y);
        g2 = filter([0 1],[1 -th(1)], u);
        r  = 2*(y-yt);
        th = th - al*[r.'*g1; r.'*g2];
        if ~all(isfinite(th)) || abs(th(1)) > 3, break; end
    end
    prt('  GD    alpha = %-9g iterations = %s\n', al, num2str(it));
    if ~isnan(it) && (isnan(best_gd(2)) || it < best_gd(2)), best_gd = [al it]; end
end

% ---- FL-CMO iteration ----------------------------------------------------
m = N-1;  nu = N+2;  rows = (1:m).';
best_fl = [NaN NaN];
for tau = [0.1 0.2 0.4 0.6 0.8 1.0]
    xi = [th0; simy(th0(1),th0(2))];          % feasible start, free of charge
    it = NaN;
    for k = 1:5000
        a = xi(1); b = xi(2); y = xi(3:end);
        h = y(2:end) - a*y(1:end-1) - b*u(1:end-1);
        if cost(y) <= tgt && norm(h) < 1e-8, it = k; break; end
        J = zeros(m,nu);
        J(:,1) = -y(1:end-1);  J(:,2) = -u(1:end-1);
        J(sub2ind([m nu],rows,2+rows))   = -a;
        J(sub2ind([m nu],rows,3+rows))   =  1;
        gL = [0; 0; 2*(y-yt)];
        lam = (J*J.')\(K*h - J*gL);           % see the sign note in the header
        xi  = xi - tau*(gL + J.'*lam);
        if ~all(isfinite(xi)), break; end
    end
    prt('  FLCMO tau   = %-9g iterations = %s\n', tau, num2str(it));
    if ~isnan(it) && (isnan(best_fl(2)) || it < best_fl(2)), best_fl = [tau it]; end
end

prt('\n  best GD    : alpha = %g, %d iterations\n', best_gd(1), best_gd(2));
prt('  best FL-CMO: tau   = %g, %d iterations\n', best_fl(1), best_fl(2));
fclose(fid);
fprintf('\nWritten to %s\n', outfile);
