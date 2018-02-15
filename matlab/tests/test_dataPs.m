% Generate the (1) UI and (2) wall-clock computation time vectors for comparing four different methods: 
% adm, fmincon with gradient and Hessian, fimincon with gradient only, and fmincon blackbox.
% The admUI_mex executable is to be generated using Matlab 2017a.

% Data for Fig. 2 (bottom panel)
% ns=ny=nz
clear all;

addpath(genpath('../'))

% pregenerated 300 distributions
load('../../data/dataPs.mat')

nsmax = 10;
Nmax = nsmax^3;

ndist = 250;  % read only a part of the full set of distributions
c = 0;

if exist("admUI_mex")
    disp("Using admUI_mex.")
    admUI_f = @admUI_mex;
else
    admUI_f = @admUIg;
end

if exist("fmincon")
    use_fmincon = true;
    imax = 4;
else
    disp("fmincon not found.  Using only admUI.")
    use_fmincon = false;
    imax = 1;
end

for ns = 2:nsmax
    disp(["ns: ", num2str(ns)])
    ny = ns; nz = ns;
    for i = 1:ndist
        c = c+1;  % iteration count

        P = Ps(:,i,ns)';
        P = P(P~=0);
        Pzys = reshape(P,nz,ny,ns);   
        Psy = squeeze(sum(Pzys,1))';
        Psz = squeeze(sum(Pzys,2))';
        
        % adm mex
        TSTART = tic;
        [ui,~] = admUI_f(Psy, Psz); 
        t = toc(TSTART);
        elapsedTime(i,ns,1) = t;
        UI(i,ns,1) = ui;

        if use_fmincon
            % fmincon with gradient and Hessian 
            TSTART = tic;
            [ui] = fn_UI_fmincon(P,nz,ns,ns,2);
            t = toc(TSTART);
            elapsedTime(i,ns,2) = t;
            UI(i,ns,2) = ui;

            % fmincon with gradient 
            TSTART = tic;
            [ui] = fn_UI_fmincon(P,nz,ns,ns,1);
            t = toc(TSTART);
            elapsedTime(i,ns,3) = t;
            UI(i,ns,3) = ui;

            % fmincon blackbox 
            TSTART = tic;
            [ui] = fn_UI_fmincon(P,nz,ns,ns,0);
            t = toc(TSTART);
            elapsedTime(i,ns,4) = t;
            UI(i,ns,4) = ui;
        end
    end

    for i = 1:imax
        melapsedTime(ns,i) = sum(elapsedTime(:,ns,i))/ndist;
        mUI(ns,i) = sum(UI(:,ns,i))/ndist;
    end
end

% rows index the different methods (each row is plotted against ny); different such plots for the different methods
mean_UIvs = mUI';            
mean_UIvs(:,1) = []
mean_timevs = melapsedTime'; 
mean_timevs(:,1) = []

