function [UI,Q,iter_outer,iter_inner] = admUI_accn(Psy, Psz, accu, g)
% admUI_accn adds acceleration capabilities to the original admUI for the I-projection step. 
% Currently the default acceleration is tested only for the case when S and either Y or Z is binary-valued.
% The plain admUI is obtained for gamma=1, eps = 1e-6.

% Inputs: Psy is a table of size S x Y and Psz is a table of size S x Z; acc is the desired accuracy (no of decimal places); g = acceleration parameter (0 < g <=1)
% Outputs: UI is the unique information (natural logarithm) and Q is the minimizing distribution, which is a table of size S x Y x Z
% Guido Montufar, 12 May 2017
% Pradeep Banerjee, 8 Jan, 2018 (added acceleration)

    if nargin < 3
        eps = 1e-6;
	g = 1;
    else
        eps = 10^(-accu);
    end
  
    S = size(Psy,1); 
    Y = size(Psy,2); 
    Z = size(Psz,2); 

    iter_outer = 1;
    iter_inner_t = 0;
    iter_inner = 0;
    
    % make strictly positive
    if 1==1 
        %eps=1e-6;
        if min(Psy(:))<= eps*1/(100*S*Y*Z)
            Psy=Psy + eps*1/(100*S*Y*Z); 
            Psy=Psy/sum(Psy(:));
        end
        if min(Psz(:))<= eps*1/(100*S*Y*Z)
            Psz=Psz + eps*1/(100*S*Y*Z); 
            Psz=Psz/sum(Psz(:));
        end
    end
    
    Ps = sum(Psy,2); 

    % initialization
    Ryz = sum(Psy,1)' * sum(Psz,1); 
    oldQyz_s = ones(S,Y,Z)/(S*Y*Z); 
    %oldQyz_s = coder.nullcopy(ones(S,Y,Z)/(S*Y*Z)); 
    Qyz_s = oldQyz_s;
    Q = oldQyz_s;
    
    % stopping criterion
    eps2 = eps / (20*S);

    % maximum number of iterations
    maxiter = 1000; 

    % matrix A (normalized) describing the linear family Delta_{P,s} \subset Delta_{YZ}
    [A] = Agenlocal(S,Y,Z);
    A = A./sum(A,2);
    sAn = svd(A);
    
    % optimal acceleration parameter gamma in Eqn 18 of the arxiv draft 
    % maximum convergence speed is achieved for gamma = second largest singular value of A.
    % Currently this is tested only for the case when at least one of Y or Z is binary-valued.
    if nargin < 4
        gamma = sAn(2);
    else
        gamma = g;
    end
    
    % optimization loop 
    for i = 1:maxiter
        for s = 1:S
            [Qyz_s(s,:,:),iterIproj] = Iproj(Psy,Psz,s,Ryz,eps2,gamma); 
	    iter_inner_t=iter_inner_t+iterIproj;
        end
	%iter_inner_t=iter_inner_t+iterIproj;
        Ryz = reshape(Ps'* reshape(Qyz_s,S,[]),Y,Z);
        
        if max(log(Qyz_s(:) ./oldQyz_s(:) )) <= eps
            % disp('converged outer')
	    iter_outer=i;
            break
        else
            % max(log(Qyz_s(:) ./oldQyz_s(:) ))
            oldQyz_s = Qyz_s; 
	    iter_outer = iter_outer+1;
        end
        if i==maxiter
	    iter_outer=i;
            disp('maxiter reached outer')
        end
    end % optimization loop
    
    % avg I-proj iterations
    iter_inner = iter_inner_t/iter_outer;
    
    % put together the minimizing joint distribution 
    for s = 1:S
        Q(s,:,:) = Ps(s) * Qyz_s(s,:,:);
    end
    % evaluate the unique information 
    UI = MIsy_z(Q)/log(2); 
end

function [Qyzs,iterIproj] = Iproj(Psy,Psz,s,Ryz,eps,gamma)
% Iproj computes argmin D(Qyzs || Ryz) over Qyzs \in Delta_{P,s} \subset Delta_{YZ}
    
    % initialization
    b = Ryz; 
    oldb = ones(size(Ryz))/numel(Ryz); 

    % for stopping criterion
    %eps = .001*eps; 

    %maxiter = 15000;
    maxiter = 1000;
    iterIproj = 0;
                                  
    al1 = (Psy(s,:)/sum(Psy(s,:)))';
    al2 = (Psz(s,:)/sum(Psz(s,:))); 

    % optimization loop
    for i = 1:maxiter

        b = b .* ((al1./sum(b,2)).^(.5/gamma) * (al2./sum(b,1)).^(.5/gamma));
        % b = b/sum(b(:));
   
        % check convergence
        if max((b(:) - oldb(:)).^2) <= eps^2
            %disp('converged')
            iterIproj = i;
            break
        else 
            oldb = b; 
        end
         
        %if i==maxiter
        %    disp('maxiter reached inner')
        %end
    end % optimization loop
    
    Qyzs = b;
end

function [I] = MIsy_z(Q)
    Qz = squeeze(sum(sum(Q,1),2));
    Z = numel(Qz);
    I = 0; 
    for z = 1:Z
        Qsyz = squeeze(Q(:,:,z)); Qsyz = Qsyz/sum(Qsyz(:)); 
        J = Qsyz .* log(Qsyz ./ ( sum(Qsyz,2) * sum(Qsyz,1) ));
        I = I + Qz(z) * sum(J(find(Qsyz)));
    end
end

function [A] = Agenlocal(ns,ny,nz)
    rsy = [ones(1,nz) zeros(1,nz*(ny-1))];
    Msy = rsy;
    for i=1:ny-1
       Msy = [Msy; circshift(rsy,i*nz)];
    end
    
    rsz = zeros(1,nz*ny);
    for i=1:nz*ny
        if mod(i,nz)==1
           rsz(i) = 1;
        end;
    end
    Msz = rsz;
    for i=1:nz-1
       t = [zeros(1,i) rsz(1:end-i)];
       Msz = [Msz; t];     
    end
    
    A = [Msy; Msz];
    
    %D = rref(null(A)');
    %A = rref(null(D)');
end

