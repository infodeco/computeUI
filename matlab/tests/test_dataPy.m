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
       c = c+1;  % iteration count

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

       % fmincon with gradient and Hessian 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,2);
       t = toc(TSTART);
       elapsedTime(i,ny,2) = t;
       UI(i,ny,2)=ui;

       % fmincon with gradient 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,1);
       t = toc(TSTART);
       elapsedTime(i,ny,3) = t;
       UI(i,ny,3)=ui;

       % fmincon blackbox 
       TSTART = tic;
       [ui] = fn_UI_fmincon(P,nz,ny,ns,0);
       t = toc(TSTART);
       elapsedTime(i,ny,4) = t;
       UI(i,ny,4)=ui;
   end

   for i=1:4
      melapsedTime(ny,i) = sum(elapsedTime(:,ny,i))/ndist;
      mUI(ny,i) = sum(UI(:,ny,i))/ndist;
   end
end

%UI(:,1,:)=[];
%elapsedTime(:,1,:)=[];
% rows index the different methods (each row is plotted against ny); different such plots for the different methods
mean_UIvy = mUI';            
mean_UIvy(:,1) = []
mean_timevy = melapsedTime'; 
mean_timevy(:,1) = []

