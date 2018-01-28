ns = 3; ny = 3; nz = 3;
N = ns*ny*nz;
P = zeros(N);
P = sort([0, rand(1,N-1), 1]); 
P = P(end:-1:2)-P(end-1:-1:1); 

P = P/sum(P);
Pzys = reshape(P,nz,ny,ns);   
Psy = squeeze(sum(Pzys,1))';
Psz = squeeze(sum(Pzys,2))';

maxiter = 1000; 
T = 1;
innerstop = 1; 
eps = 6;                   
[UI,~]=admUIg(Psy, Psz, T, innerstop, eps, maxiter) 

%% test the generated MEX
%[UI,~]=admUIg_mex(Psy, Psz, T, innerstop, eps, maxiter) 

