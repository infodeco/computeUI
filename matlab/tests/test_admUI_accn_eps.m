% Compare original admUI with admUI with acceleration; 
% Generate data for Fig. 4 in paper
clear all;

addpath(genpath('../'))

% pregenerated 300 distributions
load('../../data/dataPy.mat')

ns = 2; nz = 2 ; ny = 2;

% read only a part of the full set of distributions
ndist=250;  

neps = linspace(1,12,12);
eps = 10.^(-neps);

for g=1:12
    accu = neps(g);
    epsilon = eps(g)
    for i=1:ndist
        P = Py(:,i,ny)';
        P = P(P~=0);
        Pzys = reshape(P,nz,ny,ns);   
        Psy = squeeze(sum(Pzys,1))';
        Psz = squeeze(sum(Pzys,2))';
        
        % adm 
        TSTART = tic;
	gamma = 1;
        [ui,~,iter_outer,iter_inner_total]=admUI_accn(Psy, Psz, accu, gamma); 
        t = toc(TSTART);
        elapsedTime(i,g,1) = t;
        UI(i,g,1)=ui;
	UI_adm=ui;
        elapsedTime_adm = t;
        iter_outer(i,g,1) = iter_outer;
        iter_inner(i,g,1) = iter_inner_total;

	% adm w acceleration 
        TSTART = tic;
        [ui,~,iter_outer_acc,iter_inner_total_acc]=admUI_accn(Psy, Psz, accu); 
        t = toc(TSTART);
        elapsedTime(i,g,2) = t;
        UI(i,g,2)=ui;
	UI_adm_acc=ui;
        elapsedTime_adm_acc = t;
        iter_outer(i,g,2) = iter_outer_acc;
        iter_inner(i,g,2) = iter_inner_total_acc;
    end

    for k=1:2
        mUI(g,k) = sum(UI(:,g,k))/ndist;
        melapsedTime(g,k) = sum(elapsedTime(:,g,k))/ndist;
        miter_outer(g,k) = sum(iter_outer(:,g,k))/ndist;
        miter_inner(g,k) = sum(iter_inner(:,g,k))/ndist;
    end
end

% eps
mean_UIvy = mUI'           
mean_timevy = melapsedTime';
mean_iter_outer = miter_outer'; 
mean_iter_inner = miter_inner'

 

