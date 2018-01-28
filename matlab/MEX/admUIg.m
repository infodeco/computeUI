function [UI,Q] = admUIg(Psy, Psz, T, innerstop, accu, maxiter)
%function [UI,Q] = admUIg(Psy, Psz, conf)
%#codegen
% admUIg computes the Unique Information for the marginals P_SY and P_SZ
% Inputs: Psy is a table of size S x Y and Psz is a table of size S x Z
%         T = 1;                        % freq test stop criterion inner loop; Setting this to 20 saves 10% of computation time (double: 1 x 1); Setting this to 1 evaluates stop criterion in every iteration
%         innerstop = 1;                % stopping criterion; 1: heuristic, 2: rigorous
%         accu = 6;                     % number of decimal places up to which accuracy is desired  (double: 1 x 1)
%         maxiter = 1000;               % maximum number of iterations (double: 1 x 1) 

% Outputs: UI is the unique information (natural logarithm) and Q is the minimizing distribution, which is a table of size S x Y x Z
% Guido Montufar, 12 May 2017

    S = size(Psy,1); 
    Y = size(Psy,2); 
    Z = size(Psz,2); 
    
    %if nargin == 2
    %    conf.T = 1;                        
    %    conf.innerstop = 1;                
    %    conf.accu = 6;                    
    %    conf.maxiter = 1000;              
    %end
    if nargin == 2
        T = 1;                        % freq test stop criterion inner loop; Setting this to 20 saves 10% of computation time. (range: 1 to 100) 
        innerstop = 2;                % stopping criterion; 1: no theory, 2: from theory (range: 1, 2)
        accu = 6;                     % ; % accuracy (range: 1 to 8)
        maxiter = 1000;               % maximum number of iterations (range: 10000 to 100000) 
    end
    
    ceps = 10^(-accu);
    
    % make strictly positive
    if 1==1 
        if min(Psy(:))<= ceps / (100*S*Y*Z);
            Psy=Psy + ceps/ (100*S*Y*Z);
            Psy=Psy/sum(Psy(:));
        end
        if min(Psz(:))<= ceps/ (100*S*Y*Z); 
            Psz=Psz + ceps/ (100*S*Y*Z); 
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
    
    % optimization loop 
    for i = 1:maxiter
        for s = 1:S
            %Qyz_s(s,:,:) = Iproj(Psy,Psz,s,Ryz,conf); 
            Qyz_s(s,:,:) = Iproj(Psy,Psz,s,Ryz,T,innerstop,accu,maxiter); 
        end
        Ryz = reshape(Ps'* reshape(Qyz_s,S,[]),Y,Z);
        
        if max(log(Qyz_s(:) ./ oldQyz_s(:) )) <= ceps 
            % disp('converged outer')
            % plot(lear); drawnow
            break
        else
            % lear(i) = max(log(Qyz_s(:) ./oldQyz_s(:) )); % track learning
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

%%
function [Qyzs] = Iproj(Psy,Psz,s,Ryz,T,innerstop,accu,maxiter)
%function [Qyzs] = Iproj(Psy,Psz,s,Ryz,conf)
%#codegen
% Iproj computes argmin D(Qyzs || Ryz) over Qyzs \in Delta_P,s \subset Delta_YZ

    ceps = 10^(-accu);
    
    % initialization
    b = Ryz; 
    oldb = ones(size(Ryz))/numel(Ryz); 
    etatarget = [Psy(s,2:end)/sum(Psy(s,:)),Psz(s,2:end)/sum(Psz(s,:))]'; 
    
    al1 = (Psy(s,:)/sum(Psy(s,:)))';
    al2 = (Psz(s,:)/sum(Psz(s,:))); 
    
    % optimization loop
    for i = 1:maxiter
        b = b .* ((al1./sum(b,2)).^.5 * (al2./sum(b,1)).^.5);
        % b = b/sum(b(:)); 
    
        switch innerstop
            case 1 % stopping criterion 1 (heuristic) 
                if mod(i,T)==0 && max( abs(b(:)- oldb(:)) ) <= ceps 
                  break
                else 
                  oldb = b;
                end
    
            case 2 % stopping criterion 2 (rigorous)
                if mod(i,T)==0 && sum( abs( [sum(b(2:end,:),2); sum(b(:,2:end),1)']/sum(b(:)) - etatarget(:) ) ./ (min(b(:))/sum(b(:))) ) <= ceps/12   
                  break
                end
    
            case 3 % another stopping criteria that works well in practice
                if mod(i,T)==0 && max( (b(:)- oldb(:)).^2 ) <= ceps^2
                break
            else 
                oldb = b; 
            end
        end 
    
        % stop if maxiter reached     
        if i==maxiter
            disp('maxiter reached inner')  
        end
    end % optimization loop
    
    % return solution   
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
