function land_glob

global g t ml k ko Fmax

g=9.8135  % gravity acceleration [m/s2]
t=1; % the simulation time step [s]
ml=600 % own mass of the lander [kg]

tmax=200; % the time in which the engine can work with full power [s]
mp0=1000; % initial amount of fuel [kg]
Fmax=16000 % maximum force [N]
k=mp0/(Fmax*tmax);
ko=1/k;





