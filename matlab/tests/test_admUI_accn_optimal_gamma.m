% Test admUI acceleration (currently tested for the case when S and at least one of Y or Z are binary-valued) 
clear all;

addpath(genpath('../'))

ns = 2; nz = 2 ; ny = 2;
N = ns*ny*nz;

% partition [0,1] into N pieces
P = sort([0, rand(1,N-1), 1]); 
% random P, sampled uniformly
P = P(end:-1:2)-P(end-1:-1:1); 
P = P/sum(P);
Pzys = reshape(P,nz,ny,ns);   
Psy = squeeze(sum(Pzys,1))';
Psz = squeeze(sum(Pzys,2))';


% adm original (without acceleration)
gamma = 1;
accu = 6;
[ui,~,iter_outer,iter_inner_total]=admUI_accn(Psy, Psz, accu, gamma) 
        
% adm with acceleration 
[ui_acc,~,iter_outer_acc,iter_inner_total_acc]=admUI_accn(Psy, Psz)



