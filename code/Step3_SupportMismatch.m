%% Part 3: Performance Analysis & Benchmarking of Multi-User MMV-BOOMP
% Evaluates Joint User Scaling, Pilot Overhead Reduction, and Support Mismatch.
clear; clc; close all;
rng(2026);

%% Parameters
M = 8;              % Base Station antennas
N = 256;            % OFDM subcarriers
L = 60;             % Channel tap length
S = 12;             % Channel sparsity
SNR_dB = 0:5:30;    % SNR range in dB
num_trials = 100;   % Realisations per SNR point

fprintf('============================================================\n');
fprintf('PART 3: Multi-User MMV-BOOMP Benchmarking Suite\n');
fprintf('============================================================\n');

%% ---- Table I: Frequency-Orthogonal Pilots (Qi et al., 2015) ----
pilot_cells_16 = cell(8,1);
pilot_cells_16{1} = [8, 40, 48, 52, 72, 82, 99, 142, 145, 154, 158, 161, 183, 209, 212, 230];
pilot_cells_16{2} = [9, 41, 49, 53, 73, 83, 100, 143, 146, 155, 159, 162, 184, 210, 213, 231];
pilot_cells_16{3} = [10, 42, 50, 54, 74, 84, 101, 144, 147, 156, 160, 163, 185, 211, 214, 232];
pilot_cells_16{4} = [17, 25, 47, 56, 59, 63, 75, 111, 115, 130, 141, 149, 153, 174, 200, 250];
pilot_cells_16{5} = [12, 34, 55, 64, 67, 109, 112, 148, 173, 215, 222, 233, 238, 241, 249, 252];
pilot_cells_16{6} = [2, 15, 45, 58, 62, 66, 96, 103, 107, 132, 165, 181, 186, 189, 204, 206];
pilot_cells_16{7} = [18, 22, 33, 68, 76, 80, 88, 91, 95, 116, 133, 167, 198, 205, 229, 246];
pilot_cells_16{8} = [7, 79, 92, 117, 120, 152, 168, 180, 187, 197, 219, 223, 239, 243, 251, 255];

F_full = dftmtx(N);
F = F_full(:, 1:L);

rng(42);
pilots_16 = exp(1j * randi([0,3], M, 16) * pi/2);

A_cell_16 = cell(M, 1);
for i = 1:M
    D_i = diag(pilots_16(i, :));
    A_cell_16{i} = D_i * F(pilot_cells_16{i}, :);
end

B_blocks_16 = cell(L, 1);
for j = 1:L
    col_j = cell(M, 1);
    for i = 1:M
        col_j{i} = A_cell_16{i}(:, j);
    end
    B_blocks_16{j} = blkdiag(col_j{:});
end

%% =========================================================================
%% EXPERIMENT 1: Multi-User Joint Scaling (U = 1 vs U = 2 vs U = 4)
%% =========================================================================
fprintf('\n--- Running Experiment 1: User Scaling Analysis ---\n');

MSE_u1 = zeros(length(SNR_dB), 1);
MSE_u2 = zeros(length(SNR_dB), 1);
MSE_u4 = zeros(length(SNR_dB), 1);

for snr_idx = 1:length(SNR_dB)
    snr = SNR_dB(snr_idx);
    sigma = sqrt(10^(-snr/10));
    
    sum_u1 = 0; sum_u2 = 0; sum_u4 = 0;
    
    for trial = 1:num_trials
        common_support = sort(randperm(L, S));
        
        H_all = zeros(L, M, 4);
        for u = 1:4
            for i = 1:M
                H_all(common_support, i, u) = (randn(S,1) + 1j*randn(S,1)) / sqrt(2);
            end
        end
        
        y_cell = cell(4, 1);
        for u = 1:4
            y_u = cell(M, 1);
            for i = 1:M
                y_u{i} = A_cell_16{i} * H_all(:, i, u) + sigma * (randn(16,1) + 1j*randn(16,1)) / sqrt(2);
            end
            y_cell{u} = y_u;
        end
        
        % U = 1 Baseline (Single-User BOOMP)
        H_hat_u1 = boomp_single(A_cell_16, y_cell{1}, B_blocks_16, S);
        sum_u1 = sum_u1 + norm(H_all(:,:,1) - H_hat_u1, 'fro')^2 / norm(H_all(:,:,1), 'fro')^2;
        
        % U = 2 MMV-BOOMP
        H_hat_u2 = mmv_boomp(y_cell(1:2), B_blocks_16, S, 2, L, M);
        sum_u2 = sum_u2 + calc_mse(H_all(:,:,1:2), H_hat_u2, 2);
        
        % U = 4 MMV-BOOMP
        H_hat_u4 = mmv_boomp(y_cell(1:4), B_blocks_16, S, 4, L, M);
        sum_u4 = sum_u4 + calc_mse(H_all(:,:,1:4), H_hat_u4, 4);
    end
    
    MSE_u1(snr_idx) = sum_u1 / num_trials;
    MSE_u2(snr_idx) = sum_u2 / num_trials;
    MSE_u4(snr_idx) = sum_u4 / num_trials;
    
    fprintf('SNR = %2d dB | U=1: %.5f | U=2: %.5f | U=4: %.6f\n', ...
            snr, MSE_u1(snr_idx), MSE_u2(snr_idx), MSE_u4(snr_idx));
end

figure;
semilogy(SNR_dB, MSE_u1, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
semilogy(SNR_dB, MSE_u2, 'g-^', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(SNR_dB, MSE_u4, 'b-s', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on; xlabel('SNR / dB'); ylabel('Average Normalized MSE');
legend('Single-User BOOMP (U=1)', 'MMV-BOOMP (U=2 users)', 'MMV-BOOMP (U=4 users)', 'Location', 'southwest');
title(sprintf('Multi-User Joint Support Scaling (BS Antennas M=%d)', M));
saveas(gcf, 'part3_user_scaling.png');

%% =========================================================================
%% EXPERIMENT 2: Partial Support Mismatch Robustness (100% vs 80% vs 60%)
%% =========================================================================
fprintf('\n--- Running Experiment 2: Support Mismatch Analysis (U=4) ---\n');

overlap_ratios = [1.0, 0.8, 0.6]; % 100%, 80%, 60% shared support
MSE_mismatch = zeros(length(SNR_dB), length(overlap_ratios));

for o_idx = 1:length(overlap_ratios)
    ratio = overlap_ratios(o_idx);
    S_shared = round(S * ratio);
    S_unique = S - S_shared;
    
    for snr_idx = 1:length(SNR_dB)
        snr = SNR_dB(snr_idx);
        sigma = sqrt(10^(-snr/10));
        
        sum_mse = 0;
        
        for trial = 1:num_trials
            all_taps = randperm(L);
            shared_taps = all_taps(1:S_shared);
            rem_taps = all_taps(S_shared+1:end);
            
            H_all = zeros(L, M, 4);
            for u = 1:4
                % Each user has shared taps + unique taps
                user_unique = rem_taps((u-1)*S_unique + 1 : u*S_unique);
                user_support = sort([shared_taps, user_unique]);
                
                for i = 1:M
                    H_all(user_support, i, u) = (randn(S,1) + 1j*randn(S,1)) / sqrt(2);
                end
            end
            
            y_cell = cell(4, 1);
            for u = 1:4
                y_u = cell(M, 1);
                for i = 1:M
                    y_u{i} = A_cell_16{i} * H_all(:, i, u) + sigma * (randn(16,1) + 1j*randn(16,1)) / sqrt(2);
                end
                y_cell{u} = y_u;
            end
            
            H_hat = mmv_boomp(y_cell, B_blocks_16, S, 4, L, M);
            sum_mse = sum_mse + calc_mse(H_all, H_hat, 4);
        end
        
        MSE_mismatch(snr_idx, o_idx) = sum_mse / num_trials;
    end
    fprintf('Overlap = %3d%% | SNR 20 dB MSE = %.6f\n', ratio*100, MSE_mismatch(5, o_idx));
end

figure;
semilogy(SNR_dB, MSE_mismatch(:,1), 'b-s', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
semilogy(SNR_dB, MSE_mismatch(:,2), 'm-d', 'LineWidth', 1.5, 'MarkerSize', 6);
semilogy(SNR_dB, MSE_mismatch(:,3), 'k-x', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on; xlabel('SNR / dB'); ylabel('Average Normalized MSE');
legend('100% Shared Support (Ideal)', '80% Shared Support', '60% Shared Support', 'Location', 'southwest');
title('MMV-BOOMP Robustness under Partial Channel Support Mismatch');
saveas(gcf, 'part3_support_mismatch.png');

fprintf('\nBenchmarking complete! Figures saved as part3_user_scaling.png and part3_support_mismatch.png\n');

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
