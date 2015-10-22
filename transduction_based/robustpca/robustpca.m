function [finalA, finalE] = robustpca(tagmatrix, lambda1, lambda2, tagCorrelation, contentCorrelation, maxIters, stopPrecision, mu)

% initialization, see Algorithm 2 [zhu2010]
[m,n] = size(tagmatrix);
X_km1_A = zeros(m,n);
X_km1_E = zeros(m,n); 
X_k_A = zeros(m,n); 
X_k_E = zeros(m,n);
b_k = 1; 
b_km1 = 1; 
mu_k = mu; %0.99*mu;
mu_bar = 1e-9*mu;

% compute the Lipschitz constant
tic;
    Lf = sqrt((4*(max(svds(mu * lambda2 * contentCorrelation))).^2) + (4*(max(svds(mu * lambda2 * tagCorrelation)).^2)) + 6);
    fprintf('Lf = %f\n', Lf);
toc;

iters = 0;
done = 0;

K_pieces = 30;
cc_pieces = cell(K_pieces,1);
out_pieces = cell(K_pieces,1);
cc_size = floor(size(contentCorrelation) / K_pieces);
for i = 1:K_pieces
    if i ~= K_pieces
        cc_pieces{i} = contentCorrelation(:, (i-1)*cc_size+1:i*cc_size);
    else
        cc_pieces{i} = contentCorrelation(:, (i-1)*cc_size+1:end);
    end
end

while ~done    
    Y_k_A = X_k_A + ((b_km1-1)/b_k)*(X_k_A-X_km1_A) ;
    Y_k_E = X_k_E + ((b_km1-1)/b_k)*(X_k_E-X_km1_E) ;
    
    tic;
    parfor i = 1:K_pieces
        out_pieces{i} = Y_k_A * cc_pieces{i};
    end
    out = cell2mat(out_pieces');

    G_k_A = Y_k_A - (1/Lf) * (mu_k * lambda2 * (out + tagCorrelation * Y_k_A) + (Y_k_A + Y_k_E - tagmatrix));
    G_k_E = Y_k_E - (1/Lf) * (Y_k_A + Y_k_E - tagmatrix);
    toc;

    tic;
    [U,S,V] = svdecon(G_k_A);
    Sdiag = diag(S);
    toc;

    tic;
    T = Sdiag - mu_k / Lf;
    T = T .* double( T > 0 );
    X_kp1_A = U * diag(T) * V';

    T = abs(G_k_E) - lambda1 * mu_k / Lf;
    T = T .* double( T > 0 );
    X_kp1_E = sign(G_k_E) .* T;
    toc;

    rankA  = sum(Sdiag > mu_k / Lf);
    normE0 = sum(sum(double(abs(X_kp1_E)>0)));
           
    b_kp1 = 0.5 * (1 + sqrt(1 + 4 * b_k * b_k));
    
    T = X_kp1_A + X_kp1_E - Y_k_A - Y_k_E;
    S_kp1_A = Lf * (Y_k_A - X_kp1_A) + T;
    S_kp1_E = Lf * (Y_k_E - X_kp1_E) + T;
    
    iters = iters+1;
    cost = norm([S_kp1_A, S_kp1_E],'fro')/(Lf * max(1,norm([X_kp1_A, X_kp1_E],'fro'))) ; 
    
    % prepare next iteration
    mu_k = max(0.9 * mu_k, mu_bar);               
    b_km1 = b_k;
    b_k = b_kp1;
    X_km1_A = X_k_A; 
    X_km1_E = X_k_E;
    X_k_A = X_kp1_A; 
    X_k_E = X_kp1_E;
        
    fprintf('Iter %d - rank(A) = %d, norm(E,0) = %d, cost = %f \n',iters,rankA,normE0,cost);       

    if cost <= stopPrecision
        done = 1;
    elseif iters >= maxIters;
        fprintf('Stop due to max iterations reached.\n');
        done = 1;
    end
end

finalA = X_k_A;
finalE = X_k_E;
