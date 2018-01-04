function [UI] = fn_UI_fmincon(P,nz,ny,nx,flag)
% flag: 0 if fmincon is blackbox, 1 if derivative is included and 2 if Hessian and derivative is included
    if flag==0
        % Blackbox fmincon
        TSTART = tic;
        [~,UI,~] = fn_UI_fmincon_main_BB(P,nz,ny,nx);
        t = toc(TSTART)
    else
        % flag =1 : fmincon with derivative only; flag=2 : fmincon with Hessian
        P = reshape(P,[nz,ny,nx]); 
        P = permute(P,[3,2,1]); 
        TSTART = tic;
        [UI,~,~,~,~,~,~,~,~] = fn_UI_fmincon_main(P,flag);
        t = toc(TSTART)
    end
end

function [p,UI,exitflag] = fn_UI_fmincon_main_BB(P,nz,ny,nx)
% Compute the unique information of P using the standard blackbox Matlab optimizer fmincon. 
% Here we minimize the conditional mutual information MI(X,Y|Z) constrained to Delta_P (the polytope of distributions with the same (X,Y) and (X,Z) marginals as P). 
% P is a probability distribution given as a row vector with entries ordered as the vectorization of the table P(Z,Y,X)
% nz, ny, nx are the number of states of Z, Y, X, respectively. 
% Guido Montufar, Nov 2016
    P = P';
    P = P/sum(P);

    %fmincon finds a constrained minimum of a function of several variables.
    %    fmincon attempts to solve problems of the form:
    %     min F(X)  
    %      X                     
    %     subject to:  A*X  <= B,   (linear inequality constraints)
    %                  Aeq*X  = Beq (linear equality constraints)
    % linear inequality constraints: A.Q <= b
    % Here Q >=0, i.e. -Q <= 0
    A = -eye(numel(P));
    b = zeros(size(P));
    % linear equality constraints: Aeq⋅Q = beq = Aeq⋅P
    % Here it just says that A(Q-P) = 0, i.e. AQ = AP, i.e., (Q-P) \element of ker(A); A is the map as defined in the paper 
    % D = kera(nx,ny,nz); % The rows of D form a basis of ker(A) 
    % Aeq = null(D)'; 
    [Aeq,D] = Agen(nx,ny,nz);
    
    beq = Aeq*P;  
    lb = [];
    ub = [];
    nonlcon = [];
    
    %fmincon_algo = 'sqp';
    fmincon_algo = 'interior-point';  
    fmincon_iterations = 10000;  
    fmincon_MFE=1000000;           
    options = optimoptions('fmincon');
    options.Algorithm = fmincon_algo; 
    options.MaxFunctionEvaluations = fmincon_MFE;     
    options.MaxIterations = fmincon_iterations; 
    options.OptimalityTolerance = 1e-6;
    options.ConstraintTolerance = 1e-8;
    options.Display = 'none';
    
    fun = @(Q)CMI(Q,nz,ny,nx);
    [p,UI,exitflag] = fmincon(fun,P,A,b,Aeq,beq,[],[],nonlcon,options);
end

function [D] = kera1(nx,ny,nz)
% The rows of D form a basis of ker(A)
% Note that the entries in each row of D add to zero
    % tangent space of the linear equality constraints
    X=1:nx; Y=1:ny; Z=1:nz; 
    
    D = []; 
    yp=1; zp=1;
    
    for x=1:nx
        for y=2:ny
            for z=2:nz
               dxyz   = kron(x==X,kron(y==Y,z==Z));    % \delta_{x,y,z}
               dxypzp = kron(x==X,kron(yp==Y,zp==Z));  % \delta_{x,y',z'}
               dxypz  = kron(x==X,kron(yp==Y,z==Z));   % \delta_{x,y',z}
               dxyzp  = kron(x==X,kron(y==Y,zp==Z));   % \delta_{x,y,z'}
    
               gxyypzzp = dxyz + dxypzp - dxypz - dxyzp; % \gamma_{x;y,y';z,z'}
    
               D = [D; gxyypzzp];  % append row gxyypzzp to D
            end
        end
    end
end

function [D,dloc] = kera2(nx,ny,nz)
% Another implementation: The rows of D form a basis of ker(A)
    D = zeros(nx*ny*nz,nx*(ny - 1)*(nz -1));
    yp=1; zp=1;
    dloc = zeros(nx*(ny - 1)*(nz - 1),3); 
    
    i = 0;
    for x = 1:nx
        for y = 1:ny
            if(y ~= 1)
                for z = 1:nz
                    if z ~= 1
                        i = i + 1;
    		    d = zeros(nx,ny,nz);
                        d(x,yp,zp) = 1; d(x,y,z) = 1; d(x,yp,z) = -1; d(x,y,zp) = -1;
    		    D(:,i) = d(:);
                        dloc(i,:) = [x,y,z];
                    end
                end
            end
        end
    end
end

function [A,D] = Agen(nx,ny,nz)
% Alternative to kera (kron issue in Matlab 2017a); 
% Both kera and Agen depend only on the cardinalities of S,Y,Z.
% The exponential family is given by the matrix A
% D = ker(A)
    rxy = [ones(1,nz) zeros(1,nz*(ny-1))];
    Mxy = rxy;
    for i=1:ny-1
        Mxy = [Mxy; circshift(rxy,i*nz)];
    end
    
    rxz = zeros(1,nz*ny);
    for i=1:nz*ny
        if mod(i,nz)==1
           rxz(i) = 1;
        end;
    end
    Mxz = rxz;
    for i=1:nz-1
        t = [zeros(1,i) rxz(1:end-i)];
        Mxz = [Mxz; t];     
    end
    
    Ay = kron(eye(nx),Mxy);
    Az = kron(eye(nx),Mxz);
    A = [Ay; Az];
    
    D = rref(null(A)');
    A = rref(null(D)');
end

function [UI,x,fval,exitflag,output,lambda,grad,hessian,x_orig] = fn_UI_fmincon_main(p,flag,initial_value)
% Compute the unique information of P using the standard blackbox Matlab optimizer fmincon, now with Gradient and Hessian options: 
% Set flag = 1 : include only gradient; flag = 2 : include gradient and Hessian
% Maik Schünemann, Pradeep Banerjee, 2017 
    y0 = 1; z0 = 1;
    [nx,ny,nz] = size(p);
    
    [D,dloc] = kera2(nx,ny,nz);
    xinit = zeros(size(D,2),1);
    
    options = optimoptions('fmincon');
    options.Display = 'off';
    options.Algorithm = 'interior-point'; 
    options.HonorBounds = true;
    options.OptimalityTolerance = 1e-6;
    options.ConstraintTolerance = 1e-8;
    %options.StepTolerance = 1e-12;
    
    f = @(x) fn_grad(x,p,D);
    options.SpecifyObjectiveGradient = true;
    if flag==2
        options.HessianFcn = @(x,~) fn_hess(x,p,D,dloc);
    end

    [x_orig,fval,exitflag,output,lambda,grad,hessian] = fmincon(f,xinit,[-D;D],[p(:);1-p(:)],[],[],[],[],[],options);
    
    x = p + reshape(D * x_orig,size(p));
    x = x + max(-min(x(:),0));
    x = x ./ sum(x(:));
    Px = permute(reshape(x,nx,ny,nz),[3,2,1]);
    UI = CMI(Px(:),nz,ny,nx);
end
    
function [v,g] = fn_grad(x,p,D)
% Gradient
    [nx,ny,nz] = size(p);
    g = zeros(nx*(ny - 1)*(nz - 1),1);
    yp=1; zp=1;
    e = 1e-8;
    i = 0;
    Q = p(:) + D*x;
    if sum(abs(min(0,Q))) > e
        v = Inf;
        g = ones(size(D,2),1).*Inf;
    else
        Q = Q + max(-min(Q(:),0));
        Q = Q./sum(Q(:));
        v = MI_X_YZ(Q,nx,ny,nz);
        Q = reshape(Q,[nx,ny,nz]);
        Qyz = squeeze(sum(Q,1));
        fng = @(x,y,z,e) log2(Q(x,y,z) + e) + log2(Q(x,yp,zp) + e) - log2(Q(x,y,zp) + e) - log2(Q(x,yp,z) + e) + log2(Qyz(y,zp) + e) + log2(Qyz(yp,z)+ e) - log2(Qyz(yp,zp) + e) - log2(Qyz(y,z) + e);
        for x=1:nx
            for y=1:ny
                if y ~= yp
                    for z=1:nz
                        if z~= zp
                            i = i + 1;
                            g(i) = real(fng(x,y,z,e));
                        end
                    end
                end
            end
        end
        g(isnan(g)) = 0;
    end
end

function [CMI] = CMI(Q,nz,ny,nx)
% Conditional mutual information I(X,Y|Z)
    Pzyx = reshape(Q,nz,ny,nx);
    Pzx = squeeze(sum(Pzyx,2));
    Pzy = squeeze(sum(Pzyx,3));
    Pz = squeeze(sum(Pzy,2)); 

    c=0; cold=0; t=0;
    for i=1:nx
        for j=1:ny
            for k=1:nz
    	        ratio = ((Pz(k)*Pzyx(k,j,i))/(Pzx(k,i)*Pzy(k,j)));
	        if isnan(ratio) | ratio<=0 | isinf(ratio)  
    	      	    t= t + 1; c = cold;
    		else
    		    c = c + Pzyx(k,j,i)*log2(ratio); cold = c;
    		end
    	    end
    	end
    end
    CMI =  c; 
end

function [CMI] = CMI_alt(Q,nz,ny,nx) % not used
% Conditional mutual information I(X,Y|Z)
    Q = reshape(Q,nz,ny,nx);
    Qz  = sum(sum(Q,3),2); % Q(Z) marginal distribution

    for z=1:nz % for each value of Z do    
        Qxy_z = squeeze(Q(z,:,:)); Qxy_z = Qxy_z/sum(sum(Qxy_z)); % Q(X,Y|Z=z) rows indexed by y, columns by x
        Qx_z  = sum(Qxy_z,1); % Q(X|Z=z) row vector 
        Qy_z  = sum(Qxy_z,2); % Q(Y|Z=z) column vector
        CE(z) = sum(sum( Qxy_z .* log2(Qxy_z ./ (Qy_z * Qx_z)) )); % MI(X,Y | Z=z)
    end
    CMI =  CE*Qz; % expectation value of MI(X,Y|Z=z) over z ~ Qz
end

function [MI] = MI_X_YZ(Q,nx,ny,nz)
% Mutual information between X and (Y,Z) 
    Q = reshape(Q,[nx,ny,nz]);
    Q_X = repmat(sum(sum(Q,3),2),[1,ny,nz]);
    Q_YZ = repmat(sum(Q,1),[nx,1,1]);
    MI = Q.*(log2(Q) - log2(Q_X) - log2(Q_YZ));
    MI(isnan(MI)) = 0; MI = real(sum(MI(:)));
end

function [v] = fn_hess(x,p,D,dloc)
% Hessian
    [nx,ny,nz] = size(p);
    yp=1; zp=1;
    eps = 0.001;
    t = 1:size(D,2);
    Q = reshape(p(:)+D*x,size(p));
    Qyz = squeeze(sum(Q,1));
    v = bsxfun(@(i,j) h(i,j,dloc,yp,zp,Q,Qyz,eps),t,t');
    v(isnan(v)) = 0; 
end

function [v] = h(i,j,dloc,y,z,Q,Qyz,eps) 
% Hessian subroutine
    di = dloc(i,:);
    dj = dloc(j,:);
    dx = bsxfun(@eq,di(1),dj(:,1));
    dyp = bsxfun(@eq,di(2),dj(:,2));
    dzp = bsxfun(@eq,di(3),dj(:,3));
    dxyp = (dx + dyp) == 2; 
    dxzp = (dx + dzp) == 2;
    dypzp = (dyp + dzp) == 2;
    dxypzp = (dxyp + dzp) == 2;
    v = dx./(Q(di(1),y,z) + eps*(Q(di(1),y,z) == 0)) - dyp./(Qyz(di(2),z) + eps*(Qyz(di(2),z) == 0)) - dzp./(Qyz(y,di(3)) + eps*(Qyz(y,di(3)) == 0)) + dxyp./(Q(di(1),di(2),z) + eps*(Q(di(1),di(2),z) == 0)) + dxzp./(Q(di(1),y,di(3)) + eps*(Q(di(1),y,di(3)) == 0)) - dypzp./(Qyz(di(2),di(3)) + eps*(Qyz(di(2),di(3)) == 0)) + dxypzp./(Q(di(1),di(2),di(3)) + eps*(Q(di(1),di(2),di(3)) == 0))  - 1./(Qyz(y,z) + eps*(Qyz(y,z) == 0)) ;
end


