%% Step 2: Multi-User Massive MIMO Scaling (U in {1, 2, 4, 8})
% Corrected: Uses Table I pilots (K=16) to avoid underdetermined matrices (MK > MS).

clear; clc; close all;
rng(2026);

%% Parameters (Qi et al., 2015)
M = 8;              % Base Station antennas
N = 256;            % OFDM subcarriers
K = 16;             % Pilot subcarriers per antenna (Table I)
L = 60;             % Channel tap length
S = 12;             % Channel sparsity level
SNR_dB = 0:5:30;    % SNR range in dB
num_trials = 100;   % Monte Carlo realizations

fprintf('============================================================\n');
fprintf('STEP 2: Multi-User Scaling (M=%d, K=%d, U in {1, 2, 4, 8})\n', M, K);
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
F = F_full(:, 1:L);

rng(42);
pilots = exp(1j * randi([0,3], M, K) * pi/2);

A_cell = cell(M, 1);
for i = 1:M
    D_i = diag(pilots(i, :));
    A_cell{i} = D_i * F(pilot_cells{i}, :);
end

%% ---- Pre-construct Candidate Block Matrices B_j ----
B_blocks = cell(L, 1);
for j = 1:L
    col_j = cell(M, 1);
    for i = 1:M
        col_j{i} = A_cell{i}(:, j);
    end
    B_blocks{j} = blkdiag(col_j{:}); % Size (M*K) x M = 128 x 8
end

%% ---- Simulation Loop ----
MSE_u1 = zeros(length(SNR_dB), 1);
MSE_u2 = zeros(length(SNR_dB), 1);
MSE_u4 = zeros(length(SNR_dB), 1);
MSE_u8 = zeros(length(SNR_dB), 1);

U_max = 8;

for snr_idx = 1:length(SNR_dB)
    snr = SNR_dB(snr_idx);
    sigma = sqrt(10^(-snr/10));
    
    sum_u1 = 0; sum_u2 = 0; sum_u4 = 0; sum_u8 = 0;
    
    for trial = 1:num_trials
        % Shared support across all users and antennas
        support = sort(randperm(L, S));
        
        % True Channel Tensor H_all (L x M x U_max)
        H_all = zeros(L, M, U_max);
        for u = 1:U_max
            for i = 1:M
                H_all(support, i, u) = (randn(S,1) + 1j*randn(S,1)) / sqrt(2);
            end
        end
        
        % Received signals for all U_max users
        y_cell = cell(U_max, 1);
        for u = 1:U_max
            y_u = cell(M, 1);
            for i = 1:M
                y_u{i} = A_cell{i} * H_all(:, i, u) + sigma * (randn(K,1) + 1j*randn(K,1)) / sqrt(2);
            end
            y_cell{u} = y_u;
        end
        
        % 1. Single-User BOOMP (U=1)
        H_hat_u1 = boomp_single(A_cell, y_cell{1}, B_blocks, S);
        sum_u1 = sum_u1 + norm(H_all(:,:,1) - H_hat_u1, 'fro')^2 / norm(H_all(:,:,1), 'fro')^2;
        
        % 2. MMV-BOOMP (U=2)
        H_hat_u2 = mmv_boomp(y_cell(1:2), B_blocks, S, 2, L, M);
        sum_u2 = sum_u2 + calc_mse(H_all(:,:,1:2), H_hat_u2, 2);
        
        % 3. MMV-BOOMP (U=4)
        H_hat_u4 = mmv_boomp(y_cell(1:4), B_blocks, S, 4, L, M);
        sum_u4 = sum_u4 + calc_mse(H_all(:,:,1:4), H_hat_u4, 4);
        
        % 4. MMV-BOOMP (U=8)
        H_hat_u8 = mmv_boomp(y_cell(1:8), B_blocks, S, 8, L, M);
        sum_u8 = sum_u8 + calc_mse(H_all(:,:,1:8), H_hat_u8, 8);
    end
    
    MSE_u1(snr_idx) = sum_u1 / num_trials;
    MSE_u2(snr_idx) = sum_u2 / num_trials;
    MSE_u4(snr_idx) = sum_u4 / num_trials;
    MSE_u8(snr_idx) = sum_u8 / num_trials;
    
    fprintf('SNR = %2d dB | U=1: %.5f | U=2: %.5f | U=4: %.5f | U=8: %.6f\n', ...
            snr, MSE_u1(snr_idx), MSE_u2(snr_idx), MSE_u4(snr_idx), MSE_u8(snr_idx));
end

%% ---- Plot Figure ----
figure;
semilogy(SNR_dB, MSE_u1, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
semilogy(SNR_dB, MSE_u2, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(SNR_dB, MSE_u4, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(SNR_dB, MSE_u8, 'k-*', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on; xlabel('SNR / dB'); ylabel('Average Normalized MSE');
legend('Single-User BOOMP (U=1)', 'MMV-BOOMP (U=2)', 'MMV-BOOMP (U=4)', 'MMV-BOOMP (U=8)', 'Location', 'southwest');
title(sprintf('Multi-User Joint Support Scaling (M=%d Antennas, K=%d Pilots)', M, K));

%% ===================== Algorithmic Helper Functions =====================

function mse = calc_mse(H_true, H_hat, U)
    mse = 0;
    for u = 1:U
        mse = mse + norm(H_true(:,:,u) - H_hat(:,:,u), 'fro')^2 / norm(H_true(:,:,u), 'fro')^2;
    end
    mse = mse / U;
end

function H_hat = boomp_single(A_cell, y_cell, B_blocks, S)
    M = length(A_cell);
    L = size(A_cell{1}, 2);
    z = vertcat(y_cell{:});
    r = z; Lambda = [];
    for iter = 1:S
        corr_val = zeros(L, 1);
        for j = 1:L
            if ismember(j, Lambda); corr_val(j) = -inf; continue; end
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

function H_hat = mmv_boomp(y_cell, B_blocks, S, U, L, M)
    K = length(y_cell{1}{1});
    Z = zeros(M*K, U);
    for u = 1:U
        Z(:, u) = vertcat(y_cell{u}{:});
    end
    R = Z; Lambda = [];
    for iter = 1:S
        corr_val = zeros(L, 1);
        for j = 1:L
            if ismember(j, Lambda); corr_val(j) = -inf; continue; end
            B_j = B_blocks{j};
            V = (B_j' * B_j) \ (B_j' * R);
            corr_val(j) = norm(V, 'fro');
        end
        [~, J] = max(corr_val);
        Lambda = [Lambda, J];
        B_Lambda = cell2mat(B_blocks(Lambda).');
        X = B_Lambda \ Z;
        R = Z - B_Lambda * X;
    end
    H_hat = zeros(L, M, U);
    for u = 1:U
        for idx = 1:length(Lambda)
            j = Lambda(idx);
            H_hat(j, :, u) = X((idx-1)*M+1 : idx*M, u).';
        end
    end
end
