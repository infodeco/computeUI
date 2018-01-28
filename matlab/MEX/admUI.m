function [UI,Q] = admUI(Psy, Psz)     
%#codegen
% This the simplest version of the admUI algorithm with a very basic stopping criteria 
% that works well in practice. This file is used to generate the MEX for the tests in Fig. 2.
% In contrast, the function admUIg supports a more rigorous set of stopping criterias for the 
% inner I-projection step and is used only for the COPY example in Table 1 of the paper.

% admUI computes the Unique Information for the marginals P_SY and P_SZ
% Inputs: Psy is a table of size S x Y and Psz is a table of size S x Z
% Outputs: UI is the unique information (natural logarithm) and Q is the
% minimizing distribution, which is a table of size S x Y x Z.
% Guido Montufar, 12 May 2017
  
    S = size(Psy,1); 
    Y = size(Psy,2); 
    Z = size(Psz,2); 
    
    % make strictly positive
    if 1==1 
        epsn=1e-6;
        if min(Psy(:))<= epsn*1/(100*S*Y*Z)
            Psy=Psy + epsn*1/(100*S*Y*Z); 
            Psy=Psy/sum(Psy(:));
        end
        if min(Psz(:))<= epsn*1/(100*S*Y*Z)
            Psz=Psz + epsn*1/(100*S*Y*Z); 
            Psz=Psz/sum(Psz(:));
        end
    end
      
    Ps = sum(Psy,2); % marginal
    Ryz = sum(Psy,1)' * sum(Psz,1); % initialization
    
    eps = 1e-7;%.0001/(20*S*Y*Z); % required accuracy 
    eps2 = eps / (20*S);
  
    oldQyz_s = ones(S,Y,Z)/(S*Y*Z); 
    %oldQyz_s = coder.nullcopy(ones(S,Y,Z)/(S*Y*Z)); 
    Qyz_s = oldQyz_s;
    Q = oldQyz_s;
    maxiter = 1000; % maximum number of iterations
    
    % optimization loop 
    for i = 1:maxiter
        for s = 1:S
            Qyz_s(s,:,:) = Iproj(Psy,Psz,s,Ryz,eps2,maxiter); 
        end
        Ryz = reshape(Ps'* reshape(Qyz_s,S,[]),Y,Z);
        
        if max(log(Qyz_s(:) ./oldQyz_s(:) )) <= eps
            % disp('converged outer')
            break
        else
            % i
            % max(log(Qyz_s(:) ./oldQyz_s(:) ))
            oldQyz_s = Qyz_s; 
        end
        if i==maxiter
            disp('maxiter reached outer')
        end
    end % optimization loop
    
    % put together the minimizing joint distribution 
    for s = 1:S
        Q(s,:,:) = Ps(s) * Qyz_s(s,:,:);
    end
    % evaluate the unique information 
    UI = MIsy_z(Q)/log(2); 
end


function [Qyzs] = Iproj(Psy,Psz,s,Ryz,eps,maxiter)
%#codegen
% Iproj computes argmin D(Qyzs || Ryz) over Qyzs \in Delta_P,s \subset Delta_YZ

    b = Ryz; % initialization
    %eps = .001*eps; % for stopping criterion
    oldb = ones(size(Ryz))/numel(Ryz); 
  
    % optimization loop
    al1 = (Psy(s,:)/sum(Psy(s,:)))';
    al2 = (Psz(s,:)/sum(Psz(s,:))); 
    for i = 1:maxiter
        % b
        b = b .* ((al1./sum(b,2)).^.5 * (al2./sum(b,1)).^.5);
        
        % check convergence
        if max( (b(:)- oldb(:)).^2 ) <= eps^2
            % disp('converged')
            break
        else 
            oldb = b; 
        end
        
        if i==maxiter
            disp('maxiter reached inner')
        end
    end % optimization loop
    
    Qyzs = b;
end


function [I] = MIsy_z(Q)
%#codegen
    Qz = squeeze(sum(sum(Q,1),2));
    Z = numel(Qz);
    I = 0; 
    for z = 1:Z
        Qsyz = squeeze(Q(:,:,z)); Qsyz = Qsyz/sum(Qsyz(:)); 
        J = Qsyz .* log(Qsyz ./ ( sum(Qsyz,2) * sum(Qsyz,1) ));
        I = I + Qz(z) * sum(J(find(Qsyz)));
    end
end
