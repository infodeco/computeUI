% Generates data for Table 1 in paper
function copytest
addpath(genpath('../'))

    UIv = [];
    Errorv = [];
    Elapased_Timev = [];
    c=0;
    % ----------------------------------------
    % High dimensional COPY: UI(S;Y\Z) = I(S;Y)
    for ny=2:14
        c = c+1
        nz=ny;      
        ns=ny*nz;
        N = ns*ny*nz;
        d = ns+1;
        P = zeros(1,N);
        for i=1:ns
            P(1+(i-1)*d)=1;
        end 
        %if min(P)<= 1/(1000*ns*ny*nz)
        %    P=P + 1/(1000*ns*ny*nz);
        %end
        P = P/sum(P);
        Pzys = reshape(P,nz,ny,ns);   
        Psy = squeeze(sum(Pzys,1))';
        Psz = squeeze(sum(Pzys,2))';
        
        MI_sy  = MIxy(P,nz,ny,ns);
        fprintf('\nTrue UI = I(S;Y) = %g\n',MI_sy);
    
        T = 1;
        maxiter = 30000;
        innerstop = 2;
        % adm eps 1e-8------------------------
        eps = 8;
        TSTART_adm = tic;
        [UI_adm(1,ny),~] = admUIg_mex(Psy, Psz, T, innerstop, eps, maxiter);
        elapsedTime_adm(1,ny) = toc(TSTART_adm);
        error_adm(1,ny) = MI_sy - UI_adm(1,ny);
        
        % adm eps 1e-5------------------------
        eps = 5;
        TSTART_adm = tic;
        [UI_adm(2,ny),~] = admUIg_mex(Psy, Psz, T, innerstop, eps, maxiter);
        elapsedTime_adm(2,ny) = toc(TSTART_adm);
        error_adm(2,ny) = MI_sy - UI_adm(2,ny);
    
        % adm eps 1e-3------------------------
        eps = 3;
        TSTART_adm = tic;
        [UI_adm(3,ny),~] = admUIg_mex(Psy, Psz, T, innerstop, eps, maxiter);
        elapsedTime_adm(3,ny) = toc(TSTART_adm);
        error_adm(3,ny) = MI_sy - UI_adm(3,ny);
    
        %-------------------------------------
        innerstop = 3;
        % adm eps 1e-8------------------------
        eps = 8;
        TSTART_adm = tic;
        [UI_adm(4,ny),~] = admUIg_mex(Psy, Psz, T, innerstop, eps, maxiter);
        elapsedTime_adm(4,ny) = toc(TSTART_adm);
        error_adm(4,ny) = MI_sy - UI_adm(4,ny);
        
        % adm eps 1e-5------------------------
        eps = 5;
        TSTART_adm = tic;
        [UI_adm(5,ny),~] = admUIg_mex(Psy, Psz, T, innerstop, eps, maxiter);
        elapsedTime_adm(5,ny) = toc(TSTART_adm);
        error_adm(5,ny) = MI_sy - UI_adm(5,ny);
    
        % adm eps 1e-3------------------------
        eps = 3;
        TSTART_adm = tic;
        [UI_adm(6,ny),~] = admUIg_mex(Psy, Psz, T, innerstop, eps, maxiter);
        elapsedTime_adm(6,ny) = toc(TSTART_adm);
        error_adm(6,ny) = MI_sy - UI_adm(6,ny);
    
        %%------------------------
        % fmincon with Hessian
        P=P + 1/(100000*ns*ny*nz);
        P = P/sum(P);
        TSTART = tic;
        [UI_adm(7,ny)] = fn_UI_fmincon(P,nz,ny,ns,2);
        elapsedTime_adm(7,ny) = toc(TSTART);
        error_adm(7,ny) = MI_sy - UI_adm(7,ny);
        
    end % for ny=2:14
    
    UI_adm(:,1)=[]
    error_adm(:,1)=[]
    elapsedTime_adm(:,1)=[]
end

% compute I(X;Y)
function [mi]=MIxy(P,nz,ny,nx)
    for n=1:size(P,1) % each column is a probability distribution
        Pzyx = reshape(P(n,:),nz,ny,nx);
        Pzx = squeeze(sum(Pzyx,2));
        Pzy = squeeze(sum(Pzyx,3));
        Pyx = squeeze(sum(Pzyx,1));
        Pz = squeeze(sum(Pzy,2)); 
        Py = squeeze(sum(Pyx,2)); 
        Px = squeeze(sum(Pyx,1)); 
   
        c=0; cold=0; 
        for i=1:nx
            for j=1:ny
                ratio = (Pyx(j,i))/(Px(i)*Py(j));
                %if isnan(ratio) | ratio==0 | isinf(ratio)
                if isnan(ratio) | ratio<=0 | isinf(ratio)  % ratio==0 test is enough but sometimes matlab gives a negative value for Pzyx when reshaping when CMI o/ps imag values; so for safety ratio<=0
                    c = cold;
                else
              	c = c + Pyx(j,i)*log(ratio);
            	cold = c;
                end
           end
       end
       mi(n,1) =  c/log(2); 
    end
end
