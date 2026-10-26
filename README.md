# Massive MIMO Sparse Channel Estimation (MMV-BOOMP)

## Project Overview
This project contains the algorithmic implementation and performance benchmarking of a **Multi-User Multiple Measurement Vector Block Optimized Orthogonal Matching Pursuit (MMV-BOOMP)** algorithm. 

Developed as a self-research thesis, this project addresses the **pilot overhead bottleneck** in 5G/6G Massive MIMO wireless communication systems using Compressed Sensing (CS).

## Tools & Technologies
*   **Language:** MATLAB
*   **Domain:** Wireless Communications, Digital Signal Processing (DSP), Compressed Sensing.
*   **Key Concepts:** Massive MIMO, Sparse Channel Estimation, Orthogonal Matching Pursuit (OMP), Block-Sparsity, Multi-User Detection.

## System Model & Problem Statement
In Frequency Division Duplexing (FDD) Massive MIMO, acquiring Channel State Information (CSI) requires pilot sequences. Traditional methods scale pilot overhead linearly with the number of Base Station (BS) antennas (\(M\)), which is inefficient.

By exploiting the inherent **delay-domain sparsity** (\(S \ll L\)) and the spatial correlation of multipath reflections, this project formulates the problem as a block-sparse recovery problem:
\[
 \mathbf{Z} = \mathbf{BW} + \mathbf{N} 
\]
Where \(\mathbf{Z}\) is the stacked multi-user measurement matrix, \(\mathbf{B}\) is the block dictionary, and \(\mathbf{W}\) is the sparse channel matrix.

## Repository Structure
*   **`/src/Step1_SingleUser_BOOMP.m`**: Reproduces the baseline (Qi et al., 2015), demonstrating that joint processing across \(M\) antennas reduces block coherence and outperforms individual OMP.
*   **`/src/Step2_MultiUser_UserScaling.m`**: The proposed MMV-BOOMP framework. Pools measurements across \(U\) active users to jointly recover their shared delay support.
*   **`/src/Step3_SupportMismatch.m`**: Robustness analysis simulating realistic scenarios where users share only 60% or 80% of their channel taps.
*   **`/src/Step4_RandomSupport_OMP.m`**: Monte Carlo benchmarking against standard OMP under random sparsity realizations.
*   **`/report/`**: Contains the full thesis report detailing the mathematical derivations and results.

## Key Results
*   **86.76% Reduction in MSE:** At 20 dB SNR, the proposed multi-user MMV-BOOMP achieves an 86% reduction in Normalized Mean Squared Error (NMSE) compared to single-user methods, *without* increasing pilot overhead.
*   **Robustness:** The algorithm maintained high accuracy even when users only shared 60% of their channel support, proving its viability for real-world cellular networks.
