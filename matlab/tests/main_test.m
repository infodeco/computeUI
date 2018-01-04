% Generate the (1) UI and (2) wall-clock computation time vectors for comparing four different methods: 
% adm, fmincon with gradient and Hessian, fimincon with gradient only, and fmincon blackbox.
% The admUI_mex executable is generated using Matlab 2017a.
clear all;

% pregenerated 300 distributions
load('dataPy.mat')

ns = 2; nz = 2;
nymax = 10;
Nmax = ns*nymax*nz;

ndist=250;  % read only a part of the full set of distributions
c=0;

for ny=2:nymax
   for i=1:ndist
       c = c+1  % iteration count

       P = Py(:,i,ny)';
       P = P(P~=0);
       Pzys = reshape(P,nz,ny,ns);   
       Psy = squeeze(sum(Pzys,1))';
       Psz = squeeze(sum(Pzys,2))';
       
       % adm mex
       TSTART = tic;
       [ui,~]=admUI_mex(Psy, Psz); 
       t = toc(TSTART);
       elapsedTime(i,ny,1) = t;
       UI(i,ny,1)=ui;
       melapsedTime(ny,1) = sum(elapsedTime(:,ny,1))/ndist;
       mUI(ny,1) = sum(UI(:,ny,1))/ndist;

       % fmincon with gradient and Hessian 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,2);
       t = toc(TSTART);
       elapsedTime(i,ny,2) = t;
       UI(i,ny,2)=ui;
       melapsedTime(ny,2) = sum(elapsedTime(:,ny,2))/ndist;
       mUI(ny,2) = sum(UI(:,ny,2))/ndist;

       % fmincon with gradient 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,1);
       t = toc(TSTART);
       elapsedTime(i,ny,3) = t;
       UI(i,ny,3)=ui;
       melapsedTime(ny,3) = sum(elapsedTime(:,ny,3))/ndist;
       mUI(ny,3) = sum(UI(:,ny,3))/ndist;

       % fmincon blackbox 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,0);
       t = toc(TSTART);
       elapsedTime(i,ny,4) = t;
       UI(i,ny,4)=ui;
       melapsedTime(ny,4) = sum(elapsedTime(:,ny,4))/ndist;
       mUI(ny,4) = sum(UI(:,ny,4))/ndist;
   end
end

%UI(:,1,:)=[];
%elapsedTime(:,1,:)=[];
% rows index the different methods (each row is plotted against ny); different such plots for the different methods
mean_UIvy = mUI';            
mean_UIvy(:,1) = []
mean_timevy = melapsedTime'; 
mean_timevy(:,1) = []


%%------------------------------------------------------------------------------
%% Repeat for Pz
clear all;

% pregenerated 300 distributions
load('dataPz.mat')

ns = 2; ny = 2;
nzmax = 10;
Nmax = ns*ny*nzmax;

ndist=250;  % read only a part of the full set of distributions
c=0;
for nz=2:nzmax
   for i=1:ndist
       c = c+1  % iteration count

       P = Pz(:,i,nz)';
       P = P(P~=0);
       Pzys = reshape(P,nz,ny,ns);   
       Psy = squeeze(sum(Pzys,1))';
       Psz = squeeze(sum(Pzys,2))';
       
       % adm mex 
       TSTART = tic;
       [ui,~]=admUI_mex(Psy, Psz); 
       t = toc(TSTART);
       elapsedTime(i,nz,1) = t;
       UI(i,nz,1)=ui;
       melapsedTime(nz,1) = sum(elapsedTime(:,nz,1))/ndist;
       mUI(nz,1) = sum(UI(:,nz,1))/ndist;

       % fmincon with gradient and Hessian 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,2);
       t = toc(TSTART);
       elapsedTime(i,nz,2) = t;
       UI(i,nz,2)=ui;
       melapsedTime(nz,2) = sum(elapsedTime(:,nz,2))/ndist;
       mUI(nz,2) = sum(UI(:,nz,2))/ndist;

       % fmincon with gradient 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,1);
       t = toc(TSTART);
       elapsedTime(i,nz,3) = t;
       UI(i,nz,3)=ui;
       melapsedTime(nz,3) = sum(elapsedTime(:,nz,3))/ndist;
       mUI(nz,3) = sum(UI(:,nz,3))/ndist;

       % fmincon blackbox 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,0);
       t = toc(TSTART);
       elapsedTime(i,nz,4) = t;
       UI(i,nz,4)=ui;
       melapsedTime(nz,4) = sum(elapsedTime(:,nz,4))/ndist;
       mUI(nz,4) = sum(UI(:,nz,4))/ndist;
   end
end

% rows index the different methods (each row is plotted against ny); different such plots for the different methods
mean_UIvz = mUI';            
mean_UIvz(:,1) = []
mean_timevz = melapsedTime'; 
mean_timevz(:,1) = []



%------------------------------------------------------------------------------
% Repeat for Ps
% Symmetric ns=ny=nz
clear all;

% pregenerated 300 distributions
load('dataPs.mat')

nsmax = 10;
Nmax = nsmax^3;

ndist=250;  % read only a part of the full set of distributions
c=0;

for ns=2:nsmax
   ny=ns; nz=ns;
   for i=1:ndist
       c = c+1  % iteration count

       P = Ps(:,i,ns)';
       P = P(P~=0);
       Pzys = reshape(P,nz,ny,ns);   
       Psy = squeeze(sum(Pzys,1))';
       Psz = squeeze(sum(Pzys,2))';
       
       % adm mex
       TSTART = tic;
       [ui,~]=admUI_mex(Psy, Psz); 
       t = toc(TSTART);
       elapsedTime(i,ns,1) = t;
       UI(i,ns,1)=ui;
       melapsedTime(ns,1) = sum(elapsedTime(:,ns,1))/ndist;
       mUI(ns,1) = sum(UI(:,ns,1))/ndist;

       % fmincon with gradient and Hessian 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ns,ns,2);
       t = toc(TSTART);
       elapsedTime(i,ns,2) = t;
       UI(i,ns,2)=ui;
       melapsedTime(ns,2) = sum(elapsedTime(:,ns,2))/ndist;
       mUI(ns,2) = sum(UI(:,ns,2))/ndist;

       % fmincon with gradient 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ns,ns,1);
       t = toc(TSTART);
       elapsedTime(i,ns,3) = t;
       UI(i,ns,3)=ui;
       melapsedTime(ns,3) = sum(elapsedTime(:,ns,3))/ndist;
       mUI(ns,3) = sum(UI(:,ns,3))/ndist;

       % fmincon blackbox 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ns,ns,0);
       t = toc(TSTART);
       elapsedTime(i,ns,4) = t;
       UI(i,ns,4)=ui;
       melapsedTime(ns,4) = sum(elapsedTime(:,ns,4))/ndist;
       mUI(ns,4) = sum(UI(:,ns,4))/ndist;
   end
end

% rows index the different methods (each row is plotted against ny); different such plots for the different methods
mean_UIvs = mUI';            
mean_UIvs(:,1) = []
mean_timevs = melapsedTime'; 
mean_timevs(:,1) = []

