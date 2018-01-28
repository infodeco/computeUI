ns = 2; ny = 2; nz = 2;
N = ns*ny*nz;
P = zeros(N);
P = sort([0, rand(1,N-1), 1]); 
P = P(end:-1:2)-P(end-1:-1:1); 

P = P/sum(P);
Pzys = reshape(P,nz,ny,ns);   
Psy = squeeze(sum(Pzys,1))';
Psz = squeeze(sum(Pzys,2))';

[UI,~] = admUI(Psy, Psz)

%% test the generated MEX
%[UI,~]=admUI_mex(Psy, Psz, accu, g)

