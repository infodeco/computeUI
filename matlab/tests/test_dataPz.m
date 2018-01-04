% Generate the (1) UI and (2) wall-clock computation time vectors for comparing four different methods: 
% adm, fmincon with gradient and Hessian, fimincon with gradient only, and fmincon blackbox.
% The admUI_mex executable is generated using Matlab 2017a.
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
       c = c+1;  % iteration count

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

       % fmincon with gradient and Hessian 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,2);
       t = toc(TSTART);
       elapsedTime(i,nz,2) = t;
       UI(i,nz,2)=ui;

       % fmincon with gradient 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,1);
       t = toc(TSTART);
       elapsedTime(i,nz,3) = t;
       UI(i,nz,3)=ui;

       % fmincon blackbox 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,0);
       t = toc(TSTART);
       elapsedTime(i,nz,4) = t;
       UI(i,nz,4)=ui;
   end

   for i=1:4
      melapsedTime(nz,i) = sum(elapsedTime(:,nz,i))/ndist;
      mUI(nz,i) = sum(UI(:,nz,i))/ndist;
   end
end

% rows index the different methods (each row is plotted against ny); different such plots for the different methods
mean_UIvz = mUI';            
mean_UIvz(:,1) = []
mean_timevz = melapsedTime'; 
mean_timevz(:,1) = []




