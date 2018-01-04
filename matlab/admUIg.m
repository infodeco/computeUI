function [UI,Q] = admUIg(Psy, Psz, conf)
% admUI computes the Unique Information for the marginals P_SY and P_SZ
% Inputs: Psy is a table of size S x Y and Psz is a table of size S x Z
% Outputs: UI is the unique information (natural logarithm) and Q is the
% minimizing distribution, which is a table of size S x Y x Z
% Guido Montufar, 12 May 2017

    S = size(Psy,1); 
    Y = size(Psy,2); 
    Z = size(Psz,2); 
    
    if nargin == 2
        conf.T = 1;                   % freq test stop criterion inner loop
        conf.innerstop = 2;           % stopping criterion
    end
    conf.eps = 1e-6; % / (100*S*Y*Z); % accuracy 
    conf.maxiter = 25000;             % maximum number of iterations    
    
    % make strictly positive
    if 1==1 
        if min(Psy(:))<= conf.eps / (100*S*Y*Z);
            Psy=Psy + conf.eps/ (100*S*Y*Z);
            Psy=Psy/sum(Psy(:));
        end
        if min(Psz(:))<= conf.eps/ (100*S*Y*Z); 
            Psz=Psz + conf.eps/ (100*S*Y*Z); 
            Psz=Psz/sum(Psz(:));
        end
    end
      
    Ps = sum(Psy,2);                
    Ryz = sum(Psy,1)' * sum(Psz,1); % initialization
    oldQyz_s = ones(S,Y,Z)/(S*Y*Z); % initialization
    
    % optimization loop 
    for i = 1:conf.maxiter
        for s = 1:S
            Qyz_s(s,:,:) = Iproj(Psy,Psz,s,Ryz,conf); 
        end
        Ryz = reshape(Ps'* reshape(Qyz_s,S,[]),Y,Z);
        
        if max(log(Qyz_s(:) ./ oldQyz_s(:) )) <= conf.eps 
            %disp('converged outer')
            %plot(lear); drawnow
            break
        else
            %lear(i) = max(log(Qyz_s(:) ./oldQyz_s(:) )); % track learning
            oldQyz_s = Qyz_s; 
        end
        
        if i==conf.maxiter
            disp('maxiter reached outer')
        end
    end % optimization loop
    
    % put together the minimizing joint distribution 
    for s = 1:S
        Q(s,:,:) = Ps(s) * Qyz_s(s,:,:);
    end
    
    % evaluate the unique information 
    UI = MIsy_z(Q); 
end

function [Qyzs] = Iproj(Psy,Psz,s,Ryz,conf)
% Iproj computes argmin D(Qyzs || Ryz) over Qyzs \in Delta_P,s \subset Delta_YZ

    % eps = .001*eps; % for stopping criterion
    % maxiter = 15000;
    
    % initialization
    b = Ryz; 
    oldb = ones(size(Ryz))/numel(Ryz); 
    etatarget = [Psy(s,2:end)/sum(Psy(s,:)),Psz(s,2:end)/sum(Psz(s,:))]'; 
    
    al1 = (Psy(s,:)/sum(Psy(s,:)))';
    al2 = (Psz(s,:)/sum(Psz(s,:))); 
      
    % optimization loop
    for i = 1:conf.maxiter
        b = b .* ((al1./sum(b,2)).^.5 * (al2./sum(b,1)).^.5);
        % b = b/sum(b(:)); 
    
        switch conf.innerstop
            case 1 % stopping criterion 1 
                if mod(i,conf.T)==0 && max( abs(b(:)- oldb(:)) ) <= conf.eps 
                    %disp('converged')
                    break
                else 
                    oldb = b;
                end
    
            case 2 % stopping criterion 2 
                % note that eta = [sum(b(2:end,:),2)'; sum(b(:,2:end),1)] / sum(b(:)); 
                % if mod(i,conf.T)==0 && sum( abs( [sum(b(2:end,:),2); sum(b(:,2:end),1)']/sum(b(:)) - etatarget(:) ) ./ (min(b(:))/sum(b(:))) ) <= 1000 * conf.eps /12
                if mod(i,conf.T)==0 && sum( abs( [sum(b(2:end,:),2); sum(b(:,2:end),1)']/sum(b(:)) - etatarget(:) ) ./ (min(b(:))/sum(b(:))) ) <= conf.eps /12   
                    % disp('converged')
                    break
                end
        end 
    
        % stop if maxiter reached     
        if i==conf.maxiter
            disp('maxiter reached inner')  
        end
    end % optimization loop
    
    % return solution   
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
