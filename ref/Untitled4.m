clc;
clear all;
close all;
base_max = 100;
base_arr = [1:1:base_max];
for base=1:base_max
    myeps = abs(base*((base+1)/base-1)-1);
    format long e
    
    myeps_arr(base) = myeps;
    eps_arr(base)=eps;
    eps_myeps = [eps;myeps]
    err_arr(base) = myeps-eps;
end
 figure(1);
 plot(base_arr,myeps_arr);
 hold all;
 plot(base_arr,eps_arr);

%figure(2);
%plot(base_arr,err_arr)

for base=1:base_max
    myeps = abs(base*(1+1/base-1)-1);
    format long e
    
    myeps_arr(base) = myeps;
    eps_arr(base)=eps;
    base
    eps_myeps = [eps;myeps]
    err_arr(base) = myeps-eps;
end

figure(3);
plot(base_arr,err_arr)
