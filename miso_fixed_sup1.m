%% Step 1: Exact Reproduction of Qi et al. (2015) Single-User BOOMP
% Uses the exact frequency-orthogonal pilot patterns from Table I of the paper.

clear; clc; close all;
rng(2026);

%% Parameters (Qi et al., 2015)
M_total = 8;        % Total BS antennas
N = 256;            % Total OFDM subcarriers
K = 16;             % Pilot subcarriers per antenna
L = 60;             % Channel length
S = 12;             % Channel sparsity
SNR_dB = 0:5:30;    % SNR range in dB
num_trials = 100;   % Monte Carlo realizations

fprintf('============================================================\n');
fprintf('STEP 1: Single-User BOOMP (Deterministic Table I Pilots)\n');
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

% Fixed pilots to avoid random seed phase interference
rng(42);
pilots = exp(1j * randi([0,3], M_total, K) * pi/2);

A_cell = cell(M_total, 1);
for i = 1:M_total
    D_i = diag(pilots(i, :));
    A_cell{i} = D_i * F(pilot_cells{i}, :);
end

%% ---- Fixed Support Setup (Section V-A) ----
fixed_support = [2, 13, 21, 24, 29, 33, 41, 42, 43, 53, 54, 60];

MSE_fixed = struct('ind', zeros(size(SNR_dB)), ...
                   'm2',  zeros(size(SNR_dB)), ...
                   'm4',  zeros(size(SNR_dB)), ...
                   'm6',  zeros(size(SNR_dB)), ...
                   'm8',  zeros(size(SNR_dB)));

subsets_eval = [2, 4, 6, 8];

for snr_idx = 1:length(SNR_dB)
    snr = SNR_dB(snr_idx);
    sigma = sqrt(10^(-snr/10));
    
    sum_ind = 0; sum_m2 = 0; sum_m4 = 0; sum_m6 = 0; sum_m8 = 0;
    
    for trial = 1:num_trials
        % True Channel CIR (1/sqrt(2) normalized Rayleigh coefficients)
        H_true = zeros(L, M_total);
        for i = 1:M_total
            H_true(fixed_support, i) = (randn(S,1) + 1j*randn(S,1)) / sqrt(2);
        end
        
        % Received signals
        y_cell = cell(M_total, 1);
        for i = 1:M_total
            y_cell{i} = A_cell{i} * H_true(:, i) + sigma * (randn(K,1) + 1j*randn(K,1)) / sqrt(2);
        end
        
        % 1. Individual OMP (Baseline)
        H_hat_ind = zeros(L, M_total);
        for i = 1:M_total
            H_hat_ind(:, i) = omp_standard(A_cell{i}, y_cell{i}, S);
        end
        
        % Normalized MSE per individual channel estimation
        sum_ind = sum_ind + mean(sum(abs(H_true - H_hat_ind).^2, 1) ./ sum(abs(H_true).^2, 1));
        
        % 2. Joint BOOMP for Subsets
        for s_idx = 1:length(subsets_eval)
            M_sub = subsets_eval(s_idx);
            
            if M_sub == 8
                combos = 1:8;
            else
                combos = nchoosek(1:M_total, M_sub);
                if size(combos, 1) > 5
                    combos = combos(randperm(size(combos, 1), 5), :);
                end
            end
            
            mse_c = zeros(size(combos, 1), 1);
            for c = 1:size(combos, 1)
                idx = combos(c, :);
                H_hat_j = boomp_exact(A_cell(idx), y_cell(idx), S);
                
                mse_comb = 0;
                for local_i = 1:length(idx)
                    orig_i = idx(local_i);
                    mse_comb = mse_comb + norm(H_true(:, orig_i) - H_hat_j(:, local_i))^2 ...
                                         / norm(H_true(:, orig_i))^2;
                end
                mse_c(c) = mse_comb / length(idx);
            end
            
            val = mean(mse_c);
            switch M_sub
                case 2, sum_m2 = sum_m2 + val;
                case 4, sum_m4 = sum_m4 + val;
                case 6, sum_m6 = sum_m6 + val;
                case 8, sum_m8 = sum_m8 + val;
            end
        end
    end
    
    MSE_fixed.ind(snr_idx) = sum_ind / num_trials;
    MSE_fixed.m2(snr_idx)  = sum_m2  / num_trials;
    MSE_fixed.m4(snr_idx)  = sum_m4  / num_trials;
    MSE_fixed.m6(snr_idx)  = sum_m6  / num_trials;
    MSE_fixed.m8(snr_idx)  = sum_m8  / num_trials;
    
    fprintf('SNR = %2d dB | Ind OMP: %.5f | BOOMP (M=8): %.6f\n', ...
            snr, MSE_fixed.ind(snr_idx), MSE_fixed.m8(snr_idx));
end

%% ---- Plot Figure 2 (Qi et al., 2015) ----
figure;
semilogy(SNR_dB, MSE_fixed.ind, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
semilogy(SNR_dB, MSE_fixed.m2,  'b-s', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(SNR_dB, MSE_fixed.m4,  'g-^', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(SNR_dB, MSE_fixed.m6,  'm-d', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(SNR_dB, MSE_fixed.m8,  'k-*', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on; xlabel('SNR / dB'); ylabel('MSE');
legend('Individual sparse channel estimation', ...
       'Using 2 of 8 antennas', 'Using 4 of 8 antennas', ...
       'Using 6 of 8 antennas', 'Using all 8 antennas', 'Location', 'southwest');
title('Fixed support (Qi et al., 2015 Figure 2)');

%% ===================== Algorithmic Helper Functions =====================

function h_hat = omp_standard(A, y, S)
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

function H_hat = boomp_exact(A_cell, y_cell, S)
    M = length(A_cell);
    L = size(A_cell{1}, 2);
    
    z = vertcat(y_cell{:});
    
    B_blocks = cell(L, 1);
    for j = 1:L
        col_j = cell(M, 1);
        for i = 1:M
            col_j{i} = A_cell{i}(:, j);
        end
        B_blocks{j} = blkdiag(col_j{:});
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