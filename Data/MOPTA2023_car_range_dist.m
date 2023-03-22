% Demonstration code for MOPTA 2023 competition.
% Generating random numbers based on a truncated normal distribution. 

clear;
close all;

pd = makedist('Normal', 'mu', 100, 'sigma', 50);
t = truncate(pd, 20, 250);
samples = random(t, 10000, 1);
figure;
histogram(samples, 25);
