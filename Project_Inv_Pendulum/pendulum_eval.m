% inverted pendulum control - final state error calculation / evaluation

%function [kryt,part_eval]=pendulum_eval
clear;
pendulum_glob
global time_step friction g cart_mass pend_mass drw 
% initial states:
initial_states = [pi/6 0 0 0; 0 pi/6 0 0; 0 0 10 0; pi/12 pi/6 0 0];
% initialization of your controller:
% ...................................................
% ...................................................
% for instance:
% my_controller.init();

file_name = 'history.txt';
fid = fopen(file_name,'w');
fprintf(fid,"Fmax = 1000\n");
fprintf(fid,"time_step = %f\n",time_step);
fprintf(fid,"friction = %f\n", friction);
fprintf(fid,"g = %f\n",g);
fprintf(fid,"cart_weight = %f\n",cart_mass);
fprintf(fid,"pend_weight = %f\n",pend_mass);
fprintf(fid,"drw = %f\n",drw);

for j=1:4
   state = initial_states(j,:);
   state_arr(1,:)=state;
   collaps_ind = 1000;
   for i=1:1000
      % Place your controller function here .........................
      % .............................................................
      % for instance:
      %Force = my_controler.eval(state);
      Force = 1000;
      state = pendulum_model(state,Force);    % next state calculation
      state_arr(i+1,:) = state;             % adding to state array
      if abs(state(:,1)) > (pi/2)
          collaps_ind = i;
          break;
      end
      fxtab(i) = Force;
   end
   collapse_states = find(abs(state_arr(:,1)) > (pi/2));  % indexes of collapse states
   if length(collapse_states)>0
      disp('initial state: ');disp(initial_states(j,:));
      disp('pendulum collapsed down in step ');disp(collapse_states(1));
   end
   % adding punishment for collapse of pendulum 
   state_arr(:,1)=(abs(state_arr(:,1))>(pi/2)).*1e5+(abs(state_arr(:,1))<=(pi/2)).*state_arr(:,1);
   
   stantabk=state_arr.*state_arr;  % squre values of state vector params
   % evaluation function:
   part_eval(j)=sum(stantabk(:,1))+sum(stantabk(:,2))*0.25+sum(stantabk(:,3))*0.01+sum(stantabk(:,4))*0.01;
   max_distance(j)=max(abs(state_arr(:,3))); % max. odleglosci wozka
  
   % save in history
   for i=1:collaps_ind-2
      fprintf(fid,'%d  %f  %f  %f  %f  %f\n',j,state_arr(i,:),fxtab(i));
   end
     
end
fclose(fid); 
max_distance 
part_eval
evaluation=sum(part_eval)
