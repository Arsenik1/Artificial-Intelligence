function pendulum_glob
global time_step hh g friction cart_mass pend_mass drw cart_mom pend_mom cart_weight pend_weight

time_step=0.01;
hh=time_step*0.5;
g=9.8135;
friction=0.02;
cart_mass=10;
pend_mass=20;
drw=20;
cart_mom=cart_mass*drw;
pend_mom=pend_mass*drw;
cart_weight=cart_mass*g;
pend_weight=pend_mass*g;
