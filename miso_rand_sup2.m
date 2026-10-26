%% Step 1 (Part B): Qi et al. (2015) Figure 3 Reproduction
% Sparse Channel Estimation with Random Support Realisations (Avg over 1000 trials)
% Uses deterministic frequency-orthogonal pilots from Table I.

clear; clc; close all;
rng(2026);

%% Parameters (Qi et al., 2015)
M_total = 8;        % Total BS antennas
N = 256;            % Total OFDM subcarriers
K = 16;             % Pilot subcarriers per antenna
L = 60;             % Channel length
S = 12;             % Channel sparsity
SNR_dB = 0:5:30;    % SNR range in dB
num_trials = 1000;  % 1000 trials as specified in Section V-B of the paper

fprintf('============================================================\n');
fprintf('STEP 1 (Part B): BOOMP with Random Support (1000 Trials)\n');
fprintf('============================================================\n');

%% ---- Table I: Frequency-Orthogonal Pilots (Qi et al., 2015) ----
pilot_cells = cell(8,1);
pilot_cells{1} = [8, 40, 48, 52, 72, 82, 99, 142, 145, 154, 158, 161, 183, 209, 212, 230];
pilot_cells{2} = [9, 41, 49, 53, 73, 83, 100, 143, 146, 155, 159, 162, 184, 210, 213, 231];
pilot_cells{3} = [10, 42, 50, 54, 74, 84, 101, 144, 147, 156, 160, 163, 185, 211, 214, 232];
pilot_cells{4} = [17, 25, 47, 56, 59, 63, 75, 111, 115, 130, 141, 149, 153, 174, 200, 250];
pilot_cells{5} = [12, 34, 55, 64, 67, 109, 112, 148, 173, 215, 222, 233, 238, 241, 249, 252];
pilot_cells{6} = [2, 15, 45, 58, 62, 66, 96, 103, 107, 132, 165, 181, 186, 189, 204, 206];
pilot_cells{7} = [18, 22, 33, 68, 76, 80, 88, 91, 95, 116, 133, 167, 198, 205, 229, 246];
pilot_cells{8} = [7, 79, 92, 117, 120, 152, 168, 180, 187, 197, 219, 223, 239, 243, 251, 255];

F_full = dftmtx(N);
F = F_full(:, 1:L); % K x L submatrix

rng(42);
pilots = exp(1j * randi([0,3], M_total, K) * pi/2);

A_cell = cell(M_total, 1);
for i = 1:M_total
    D_i = diag(pilots(i, :));
    A_cell{i} = D_i * F(pilot_cells{i}, :);
end

%% ---- Pre-construct Candidate Block Matrices B_j ----
B_blocks = cell(L, 1);
for j = 1:L
    col_j = cell(M_total, 1);
    for i = 1:M_total
        col_j{i} = A_cell{i}(:, j);
    end
    B_blocks{j} = blkdiag(col_j{:}); % MK x M
end

%% ---- Simulation Loop (Figure 3 Evaluates Individual, M=4, and M=8) ----
MSE_rand = struct('ind', zeros(size(SNR_dB)), ...
                  'm4',  zeros(size(SNR_dB)), ...
                  'm8',  zeros(size(SNR_dB)));

for snr_idx = 1:length(SNR_dB)
    snr = SNR_dB(snr_idx);
    sigma = sqrt(10^(-snr/10));
    
    sum_ind = 0; sum_m4 = 0; sum_m8 = 0;
    
    for trial = 1:num_trials
        % Random channel support per realization (shared across M antennas)
        random_support = sort(randperm(L, S));
        
        H_true = zeros(L, M_total);
        for i = 1:M_total
            H_true(random_support, i) = (randn(S,1) + 1j*randn(S,1)) / sqrt(2);
        end
        
        y_cell = cell(M_total, 1);
        for i = 1:M_total
            y_cell{i} = A_cell{i} * H_true(:, i) + sigma * (randn(K,1) + 1j*randn(K,1)) / sqrt(2);
        end
        
        % 1. Individual OMP Baseline (M=1)
        H_hat_ind = zeros(L, M_total);
        for i = 1:M_total
            H_hat_ind(:, i) = omp_fast(A_cell{i}, y_cell{i}, S);
        end
        sum_ind = sum_ind + mean(sum(abs(H_true - H_hat_ind).^2, 1) ./ sum(abs(H_true).^2, 1));
        
        % 2. Joint BOOMP using 4 antennas (averaged over 5 combinations)
        combos_4 = nchoosek(1:M_total, 4);
        combos_4 = combos_4(randperm(size(combos_4,1), 5), :);
        
        mse_4 = 0;
        for c = 1:size(combos_4, 1)
            idx = combos_4(c, :);
            H_hat_j4 = boomp_subsets(A_cell(idx), y_cell(idx), S);
            
            mse_comb = 0;
            for local_i = 1:4
                orig_i = idx(local_i);
                mse_comb = mse_comb + norm(H_true(:, orig_i) - H_hat_j4(:, local_i))^2 ...
                                     / norm(H_true(:, orig_i))^2;
            end
            mse_4 = mse_4 + (mse_comb / 4);
        end
        sum_m4 = sum_m4 + (mse_4 / size(combos_4, 1));
        
        % 3. Joint BOOMP using all 8 antennas
        H_hat_j8 = boomp_fast_8(A_cell, y_cell, B_blocks, S);
        mse_8 = norm(H_true - H_hat_j8, 'fro')^2 / norm(H_true, 'fro')^2;
        sum_m8 = sum_m8 + mse_8;
    end
    
    MSE_rand.ind(snr_idx) = sum_ind / num_trials;
    MSE_rand.m4(snr_idx)  = sum_m4  / num_trials;
    MSE_rand.m8(snr_idx)  = sum_m8  / num_trials;
    
    fprintf('SNR = %2d dB | Ind OMP: %.5f | BOOMP (M=4): %.5f | BOOMP (M=8): %.6f\n', ...
            snr, MSE_rand.ind(snr_idx), MSE_rand.m4(snr_idx), MSE_rand.m8(snr_idx));
end

%% ---- Plot Figure 3 (Qi et al., 2015) ----
figure;
semilogy(SNR_dB, MSE_rand.ind, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
semilogy(SNR_dB, MSE_rand.m4,  'g-v', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(SNR_dB, MSE_rand.m8,  'k-*', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on; xlabel('SNR / dB'); ylabel('MSE');
legend('Individual sparse channel estimation for each BS antenna', ...
       'Using 4 of 8 antennas for joint sparse channel estimation', ...
       'Using all 8 antennas for joint sparse channel estimation', 'Location', 'southwest');
title(sprintf('Random positions of nonzero CIR entries (Avg over %d trials)', num_trials));

%% ===================== Algorithmic Helper Functions =====================

function h_hat = omp_fast(A, y, S)
    [~, L] = size(A);
    r = y;
    Lambda = [];
    for iter = 1:S
        corr = abs(A' * r);
        corr(Lambda) = -inf;
        [~, idx] = max(corr);
        Lambda = [Lambda, idx];
        A_Lambda = A(:, Lambda);
        h_temp = A_Lambda \ y;
        r = y - A_Lambda * h_temp;
    end
    h_hat = zeros(L,1);
    h_hat(Lambda) = h_temp;
end

function H_hat = boomp_fast_8(A_cell, y_cell, B_blocks, S)
    M = length(A_cell);
    L = size(A_cell{1}, 2);
    
    z = vertcat(y_cell{:});
    r = z;
    Lambda = [];
    
    for iter = 1:S
        corr_val = zeros(L, 1);
        for j = 1:L
            if ismember(j, Lambda)
                corr_val(j) = -inf;
                continue;
            end
            B_j = B_blocks{j};
            proj = (B_j' * B_j) \ (B_j' * r);
            corr_val(j) = norm(proj, 2);
        end
        
        [~, J] = max(corr_val);
        Lambda = [Lambda, J];
        
        B_Lambda = cell2mat(B_blocks(Lambda).');
        coeffs = B_Lambda \ z;
        r = z - B_Lambda * coeffs;
    end
    
    H_hat = zeros(L, M);
    for idx = 1:length(Lambda)
        j = Lambda(idx);
        H_hat(j, :) = coeffs((idx-1)*M+1 : idx*M).';
    end
end

function H_hat = boomp_subsets(A_cell, y_cell, S)
    M = length(A_cell);
    L = size(A_cell{1}, 2);
    
    z = vertcat(y_cell{:});
    
    B_sub = cell(L, 1);
    for j = 1:L
        col_j = cell(M, 1);
        for i = 1:M
            col_j{i} = A_cell{i}(:, j);
        end
        B_sub{j} = blkdiag(col_j{:});
    end
    
    r = z;
    Lambda = [];
    
    for iter = 1:S
        corr_val = zeros(L, 1);
        for j = 1:L
            if ismember(j, Lambda)
                corr_val(j) = -inf;
                continue;
            end
            B_j = B_sub{j};
            proj = (B_j' * B_j) \ (B_j' * r);
            corr_val(j) = norm(proj, 2);
        end
        
        [~, J] = max(corr_val);
        Lambda = [Lambda, J];
        
        B_Lambda = cell2mat(B_sub(Lambda).');
        coeffs = B_Lambda \ z;
        r = z - B_Lambda * coeffs;
    end
    
    H_hat = zeros(L, M);
    for idx = 1:length(Lambda)
        j = Lambda(idx);
        H_hat(j, :) = coeffs((idx-1)*M+1 : idx*M).';
    end
end