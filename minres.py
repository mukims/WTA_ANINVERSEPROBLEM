from cvxopt import matrix
from cvxopt.blas import scal, copy, dotu, nrm2, axpy
from math import sqrt

# 2009_05_11 - removed unnecessary declaration of '_y' and '_w'
# 2009_06_07 - synchronized with May 11 version hosted at Stanford
# 2009_10_26 - corrected minor error in SymOrtho,  thanks to Mridul Aanjaneya at Stanford
    
"""
a,b are scalars

On exit, returns scalars c,s,r
"""
def SymOrtho(a,b):
    aa=abs(a)
    ab=abs(b)
    if b==0.:
        s=0.
        r=aa
        if aa==0.:
            c=1.
        else:
            c=a/aa
    elif a==0.:
        c=0.
        s=b/ab
        r=ab
    elif ab>=aa:
        sb=1
        if b<0: sb=-1
        tau=a/b
        s=sb*(1+tau**2)**-0.5
        c=s*tau
        r=b/s
    elif aa>ab:
        sa=1
        if a<0: sa=-1
        tau=b/a
        c=sa*(1+tau**2)**-0.5
        s=c*tau
        r=a/c
        
    return c,s,r

"""
function [ x, istop, itn, rnorm, Arnorm, Anorm, Acond, ynorm ] = ...
           minres( A, b, M, shift, show, check, itnlim, rtol )

%        [ x, istop, itn, rnorm, Arnorm, Anorm, Acond, ynorm ] = ...
%          minres( A, b, M, shift, show, check, itnlim, rtol )
%
% minres solves the n x n system of linear equations Ax = b
% or the n x n least squares problem           min ||Ax - b||_2^2,
% where A is a symmetric matrix (possibly indefinite or singular)
% and b is a given vector.  The dimension n is defined by length(b).
%
% INPUT:
%
% "A" may be a dense or sparse matrix (preferably sparse!)
% or a function handle such that y = A(x) returns the product
% y = A*x for any given n-vector x.
%
% If M = [], preconditioning is not used.  Otherwise,
% "M" defines a positive-definite preconditioner M = C*C'.
% "M" may be a dense or sparse matrix (preferably sparse!)
% or a function handle such that y = M(x) solves the system
% My = x given any n-vector x.
%
% If shift ~= 0, minres really solves (A - shift*I)x = b
% (or the corresponding least-squares problem if shift is an
% eigenvalue of A).
%
% When M = C*C' exists, minres implicitly solves the system
%
%            P(A - shift*I)P'xbar = Pb,
%    i.e.               Abar xbar = bbar,
%    where                      P = inv(C),
%                            Abar = P(A - shift*I)P',
%                            bbar = Pb,
%
% and returns the solution      x = P'xbar.
% The associated residual is rbar = bbar - Abar xbar
%                                 = P(b - (A - shift*I)x)
%                                 = Pr.
%
% OUTPUT:
%
% x      is the final estimate of the required solution
%        after k iterations, where k is return in itn.
% istop  is a value from [-1:9] to indicate the reason for termination.
%        The reason is summarized in msg[istop+2] below.
% itn    gives the final value of k (the iteration number).
% rnorm  estimates norm(r_k)  or norm(rbar_k) if M exists.
% Arnorm estimates norm(Ar_{k-1}) or norm(Abar rbar_{k-1}) if M exists.
%        NOTE THAT Arnorm LAGS AN ITERATION BEHIND rnorm.

% Code authors:Michael Saunders, SOL, Stanford University
%              Sou Cheng Choi,  SCCM, Stanford University
%
% 02 Sep 2003: Date of Fortran 77 version, based on 
%              C. C. Paige and M. A. Saunders (1975),
%              Solution of sparse indefinite systems of linear equations,
%              SIAM J. Numer. Anal. 12(4), pp. 617-629.
%
% 02 Sep 2003: ||Ar|| now estimated as Arnorm.
% 17 Oct 2003: f77 version converted to MATLAB.
% 03 Apr 2005: A must be a matrix or a function handle.
% 10 May 2009: Parameter list shortened.
%              Documentation updated following suggestions from
%              Jeffery Kline <jeffery.kline@gmail.com>
%              (author of new Python versions of minres, symmlq, lsqr).

% % Known bugs: % 1. ynorm is currently mimicking ynorm in symmlq.m.
% It should be sqrt(x'Mx), but doesn't seem to be correct.  % Users
really want xnorm = norm(x) anyway.  It would be safer % to compute it
directly.  % 2. As Jeff Kline pointed out, Arnorm = ||A r_{k-1}|| lags
behind % rnorm = ||r_k||.  On singular systems, this means that a good
% least-squares solution exists before Arnorm is small enough % to
recognize it.  The solution x_{k-1} gets updated to x_k % (possibly a
very large solution) before Arnorm shuts things % down the next
iteration.  It would be better to keep x_{k-1}.
%------------------------------------------------------------------
"""

def minres( A, b, M=None, shift=0.0, show=False, check=False,
            itnlim=None, rtol=1e-7, eps=2.2e-16 ): 
    msg = (' beta2 = 0.  If M = I, b and x are eigenvectors '   ,
           ' beta1 = 0.  The exact solution is  x = 0       '   ,
           ' A solution to Ax = b was found, given rtol     '   ,
           ' A least-squares solution was found, given rtol '   ,
           ' Reasonable accuracy achieved, given eps        '   ,
           ' x has converged to an eigenvector              '   ,
           ' acond has exceeded 0.1/eps                     '   ,
           ' The iteration limit was reached                '   ,
           ' A  does not define a symmetric matrix          '   ,
           ' M  does not define a symmetric matrix          '   ,
           ' M  does not define a pos-def preconditioner    ')
    

    n      = len(b)
    if itnlim is None: itnlim = 5*n

    precon=True
    if M is None:        precon=False
        
    if show:
        print('\n minres.m   SOL, Stanford University   Version of 10 May 2009')
        print('\n Solution of symmetric Ax = b or (A-shift*I)x = b')
        print('\n\n n      =%8g    shift =%22.14e' % (n,shift))
        print('\n itnlim =%8g    rtol  =%10.2e\n'  % (itnlim,rtol))

    istop = 0;   itn   = 0;   Anorm = 0;    Acond = 0;
    rnorm = 0;   ynorm = 0;   done  = False;
    x     = matrix(0., (n,1))

    """
    %------------------------------------------------------------------
    % Set up y and v for the first Lanczos vector v1.
    % y  =  beta1 P' v1,  where  P = C**(-1).
    % v is really P' v1.
    %------------------------------------------------------------------
    """
    y     = +b;
    r1    = +b;
    if precon:        M(y)              # y = minresxxxM( M,b ); end
    beta1 = dotu(b, y)                  # beta1 = b'*y;

    """
    %  Test for an indefinite preconditioner.
    %  If b = 0 exactly, stop with x = 0.
    """
    if beta1< 0: istop = 8;  show = True;  done = True;
    if beta1==0:             show = True;  done = True;


    if beta1> 0:
        beta1  = sqrt( beta1 );       # Normalize y to get v1 later.
    # end if
    
    """
    % See if M is symmetric.
    """
    r2=matrix(0., (n,1))
    if check and precon:
        copy(r2, y)                     # r2     = minresxxxM( M,y );
        M(r2)
        s = nrm2(y)**2                  # s      = y' *y;
        t = dotu(r1, r2)                # t      = r1'*r2;
        z = abs(s-t)                    # z      = abs(s-t);
        epsa = (s+eps)*eps**(1./3.)     # epsa   = (s+eps)*eps^(1/3);
        if z > epsa: istop = 7;  show = True;  done = True;
    # end if
    """
    % See if A is symmetric.
    """
    w=matrix(0., (n,1))
    if check:
        A(y, w)                         # w    = minresxxxA( A,y );
        A(w, r2)                        # r2   = minresxxxA( A,w );
        s = nrm2(w)**2                  # s    = w'*w;
        t= dotu(y, r2)                  # t    = y'*r2;
        z = abs(s-t)                    # z    = abs(s-t);
        epsa = (s+eps)*eps**(1./3.)     # epsa = (s+eps)*eps^(1/3);
        if z > epsa: istop = 6;  done  = True;  show = True # end if
    # end if

    """
    %------------------------------------------------------------------
    % Initialize other quantities.
    % ------------------------------------------------------------------
    """
    oldb   = 0;       beta   = beta1;   dbar   = 0;       epsln  = 0;
    qrnorm = beta1;   phibar = beta1;   rhs1   = beta1;
    rhs2   = 0;       tnorm2 = 0;       ynorm2 = 0;
    cs     = -1;      sn     = 0;
    Arnorm = 0;
    
    scal(0., w)                         # w      = zeros(n,1);
    w2     = matrix(0., (n,1))
    copy(r1, r2)                        # r2     = r1;
    v =matrix(0., (n, 1))
    w1=matrix(0., (n,1))

    if show: 
        print(' ')
        print(' ')
        head1 = '   Itn     x[0]     Compatible    LS';
        head2 = '         norm(A)  cond(A)';
        head2 +=' gbar/|A|';  # %%%%%% Check gbar
        print(head1 + head2)
    """
    
    %---------------------------------------------------------------------
    % Main iteration loop.
    % --------------------------------------------------------------------
    """
    if not done:                     #  k = itn = 1 first time through
        while itn < itnlim: 
            itn    = itn+1;
            """
            %-----------------------------------------------------------------
            % Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
            % The general iteration is similar to the case k = 1 with v0 = 0:
            %
            %   p1      = Operator * v1  -  beta1 * v0,
            %   alpha1  = v1'p1,
            %   q2      = p2  -  alpha1 * v1,
            %   beta2^2 = q2'q2,
            %   v2      = (1/beta2) q2.
            %
            % Again, y = betak P vk,  where  P = C**(-1).
            % .... more description needed.
            %-----------------------------------------------------------------
            """
            s = 1/beta;                 # Normalize previous vector (in y).
            """
            v = s*y;                    # v = vk if P = I
            y = minresxxxA( A,v ) - shift*v;
            if itn >= 2, y = y - (beta/oldb)*r1; end
            """
            copy(y, v)
            scal(s, v)
            A(v,y)
            if abs(shift)>0:            axpy(v,y,-shift)
            if itn >= 2:                axpy(r1,y,-beta/oldb)

            alfa   = dotu(v, y)         # alphak
            axpy(r2, y, -alfa/beta)     # y    = (- alfa/beta)*r2 + y;
            
            # r1     = r2;
            # r2     = y;
            copy(y,r1)
            _y=r1
            r1=r2
            r2=y
            y=_y

            if precon:  M(y)        # y = minresxxxM( M,r2 ); # end if
            oldb   = beta;              # oldb = betak
            beta   = dotu(r2,y)         # beta = betak+1^2
            if beta < 0: istop = 6;  break # end if
            beta   = sqrt(beta)
            tnorm2 = tnorm2 + alfa**2 + oldb**2 + beta**2

            if itn==1:                  # Initialize a few things.
                if beta/beta1 < 10*eps: # beta2 = 0 or ~ 0.
                    istop = -1;         # Terminate later.
                # end if
                # %tnorm2 = alfa**2  ??
                gmax   = abs(alfa)      # alpha1
                gmin   = gmax           # alpha1
            # end if
            """
            % Apply previous rotation Qk-1 to get
            %   [deltak epslnk+1] = [cs  sn][dbark    0   ]
            %   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
            """
            oldeps = epsln
            delta  = cs*dbar + sn*alfa  # delta1 = 0         deltak
            gbar   = sn*dbar - cs*alfa  # gbar 1 = alfa1     gbar k
            epsln  =           sn*beta  # epsln2 = 0         epslnk+1
            dbar   =         - cs*beta  # dbar 2 = beta2     dbar k+1
            root   = sqrt(gbar**2 + dbar**2)
            Arnorm = phibar*root;       # ||Ar{k-1}||
            """
            % Compute the next plane rotation Qk
            gamma  = norm([gbar beta]); % gammak
            gamma  = max([gamma eps]);
            cs     = gbar/gamma;        % ck
            sn     = beta/gamma;        % sk
            """
            cs,sn,gamma=SymOrtho(gbar,beta)
            phi    = cs * phibar ;      # phik
            phibar = sn * phibar ;      # phibark+1

            """
            % Update  x.
            """
            denom = 1/gamma;
            """
            w1    = w2;
            w2    = w;
            w     = (v - oldeps*w1 - delta*w2)*denom;
            x     = x + phi*w;
            """
            copy(w, w1)
            _w=w1
            w1=w2
            w2=w
            w=_w
            
            scal(-delta,w)
            axpy(w1,    w,-oldeps)
            axpy(v,     w)
            scal(denom, w)
            axpy(w,     x, phi)
            """
            % Go round again.
            """
            gmax   = max(gmax, gamma);
            gmin   = min(gmin, gamma);
            z      = rhs1/gamma;
            # ynorm2 = z**2  + ynorm2;
            ynorm2 = nrm2(x)**2
            #rhs1   = rhs2 - delta*z;
            #rhs2   =      - epsln*z;
            """
            % Estimate various norms.
            """
            Anorm  = sqrt( tnorm2 )
            ynorm  = sqrt( ynorm2 )
            epsa   = Anorm*eps;
            epsx   = Anorm*ynorm*eps;
            epsr   = Anorm*ynorm*rtol;
            diag   = gbar;
            if diag==0: diag = epsa;    # end if

            qrnorm = phibar;
            rnorm  = qrnorm;
            test1  = rnorm/(Anorm*ynorm); #  ||r|| / (||A|| ||x||)
            test2  = root / Anorm; # ||Ar{k-1}|| / (||A|| ||r_{k-1}||)
            """
            % Estimate  cond(A).
            % In this version we look at the diagonals of  R  in the
            % factorization of the lower Hessenberg matrix,  Q * H = R,
            % where H is the tridiagonal matrix from Lanczos with one
            % extra row, beta(k+1) e_k^T.
            """
            Acond  = gmax/gmin;
            """
            % See if any of the stopping criteria are satisfied.
            % In rare cases, istop is already -1 from above (Abar = const*I).
            """
            if istop==0:
                t1 = 1 + test1;       # These tests work if rtol < eps
                t2 = 1 + test2;
                if t2    <= 1      :istop = 2; # end if 
                if t1    <= 1      :istop = 1; # end if
                if itn   >= itnlim :istop = 5; # end if
                if Acond >= 0.1/eps:istop = 4; # end if
                if epsx  >= beta1  :istop = 3; # end if
                if test2 <= rtol   :istop = 2; # end if 
                if test1 <= rtol   :istop = 1; # end if
            # end if
            """
            % See if it is time to print something.
            """
            prnt   = False;
            if n      <= 40       : prnt = True; # end if
            if itn    <= 10       : prnt = True; # end if
            if itn    >= itnlim-10: prnt = True; # end if
            if itn%10 == 0        : prnt = True  # end if
            if qrnorm <= 10*epsx  : prnt = True; # end if
            if qrnorm <= 10*epsr  : prnt = True; # end if
            if Acond  <= 1e-2/eps : prnt = True; # end if
            if istop  !=  0       : prnt = True; # end if

            if show and prnt:
                str1 = '%6g %12.5e %10.3e' % ( itn, x[0], test1 );
                str2 = ' %10.3e'           % ( test2 );
                str3 = ' %8.1e %8.1e'      % ( Anorm, Acond );
                str3 +=' %8.1e'            % ( gbar/Anorm);
                print(str1, str2, str3)
            # end if
            if abs(istop) > 0: break;        # end if
        # end while % main loop
    # end % if ~done early
    """
    % Display final status.
    """
    if show:
        print(" ")
        print(' istop   =  %3g               itn   =%5g'% (istop,itn))
        print(' Anorm   =  %12.4e      Acond =  %12.4e' % (Anorm,Acond))
        print(' rnorm   =  %12.4e      ynorm =  %12.4e' % (rnorm,ynorm))
        print(' Arnorm  =  %12.4e' % Arnorm)
        print(msg[istop+2])
    return x, istop, itn, rnorm, Arnorm, Anorm, Acond, ynorm
