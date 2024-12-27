function [U] = TVAL(pattern_num,pattern_path,intensity_path)

opts.mu = 2^12;
opts.beta = 2^8;
opts.mu0 = 2^4;       % trigger continuation shceme
opts.beta0 = 2^-5;    % trigger continuation shceme
opts.maxcnt = 10;
opts.tol_inn = 1e-16;
opts.tol = 1E-4;
opts.maxit = 2000;

A = csvread(pattern_path);
f = csvread(intensity_path);
[~,Res2] = size(A);
p = sqrt(Res2);
q = sqrt(Res2);

f_temp = f';

[U, ~] = TVAL3(A,f_temp,p,q,opts);
U = mat2gray(U);
img_path = strsplit(intensity_path,'.');
end

