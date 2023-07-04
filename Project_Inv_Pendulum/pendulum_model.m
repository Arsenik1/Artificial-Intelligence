
% Inverted pendulum state calculation
% state - state parameter vector in time t
% staten -   -||- in time t + time_step (next state)
% Fx - force from left side of the cart

function staten=pendulum_model(state,Fx)
global time_step hh g friction cart_mass pend_mass drw cart_mom pend_mom cart_weight pend_weight
%pendulum_glob
if Fx > 1000
    Fx = 1000;
elseif Fx < -1000
    Fx = -1000;
end
sx=sin(state(1));
cx=cos(state(1));
c1=cart_mass+pend_mass*sx*sx;
c2=pend_mom*state(2)*state(2)*sx;
c3=friction*state(4)*cx;
statepoch(1)=state(2);
statepoch(2)=((pend_weight+cart_weight)*sx-c2*cx+c3-Fx*cx)/(drw*c1);
statepoch(3)=state(4);
statepoch(4)=(c2-pend_weight*sx*cx-c3+Fx)/c1;

stateh=state+statepoch*hh;
  
sx=sin(stateh(1));
cx=cos(stateh(1));
c1=cart_mass+pend_mass*sx*sx;
c2=pend_mom*stateh(2)*stateh(2)*sx;
c3=friction*stateh(4)*cx;

statepochh(1)=stateh(2);
statepochh(2)=((pend_weight+cart_weight)*sx-c2*cx+c3-Fx*cx)/(drw*c1);
statepochh(3)=stateh(4);
statepochh(4)=(c2-pend_weight*sx*cx-c3+Fx)/c1;
 
staten=state+statepochh*time_step;   

if staten(1)>pi 
 staten(1)=staten(1)-2*pi;
end
if staten(1)<-pi
 staten(1)=staten(1)+2*pi;
end


