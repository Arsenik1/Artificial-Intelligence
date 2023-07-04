% LANDER SIMULATION  
land_glob;  % global variables
number_of_steps = 5000;
if_animation = 1;
show_inference = 1;
initial_states = [ 200 220 0;1000 3000 10; 400 2000 -60;800 4000 -100; 400 50 150];  % the set of initial states - each state in a row, 

                                     % each state consists of 3 parameters: [fuel amount, x, v]
number_of_initial_states = length(initial_states(:,1))                         

fis = readfis('test3.fis');                            % reading fuzzy system from file 
V_final = [];
fuel_final = [];
last_step = [];
history = [];
for i=1:number_of_initial_states                         % loop by initial states
   state = initial_states(i,:);
   x = state(2);
   step = 0;
   history = [];   
   while (step < number_of_steps)&(x~=0)                 % loop by control steps till landing   
      amount_of_fuel = state(1);
      x = state(2);
      V = state(3);
      
      F = evalfis([amount_of_fuel x V],fis);             % determination of rocket thrust
      
      %[step state F]                                    % parameter display
      
      stan_n = lander_model(state,F);           % determining a new state after a certain time interval
      state = stan_n;
      step = step+1;
            
      history(step,:) = [state F];         % record of current state + proposed thrust (it doesn't have to be because F_max (see land_glob.m)
   end
   V_final(i) = V;                         % saving of final values
   fuel_final(i) = amount_of_fuel;
   last_step(i) = step;
   
   history(:,4) = history(:,4)/100;      % chart preparation
   plot(history)
   legend('fuel','x','V','F/100');
   title(strcat('lander control (v.0) for initial atate: ',num2str(initial_states(i,:))));
   xlabel(strcat('landing speed V_land = ',num2str(V),' m/s'));
   pause
   close
   
   % Animation of lander:
   if if_animation
      history(:,4) = history(:,4).*(history(:,1)>0);  % the real force (thrust) which was used
      max_x = max(history(:,2));
      axis([-40  40  -max_x/8  max_x*1.1]);
      ylabel('height [m]');
      title(strcat('Lander animation v.0'));
      xkab = [-2 -1 -4 -3 3 4 1 2];
      ykab = [0 2 2 6 6 2 2 0]*max_x/100;
      xciag = [-2 -2 2 2];
      yciag = [-history(1,4)*max_x/1000 0 0 -history(1,4)*max_x/1000] + history(1,2);
      y = ykab+history(1,2);
      idkab = patch(xkab,y , 'b', 'erasemode','xor');     % drawing the filled hull contour
      idciag = patch(xciag,yciag, 'y','erasemode','xor');  % drawing a flame
      patch([-40 -40 40 40],[-max_x/8 0 0 -max_x/8] , 'g', 'erasemode','xor');   % drowing a ground
      %idl=line(xs, ys, 'erasemode','xor')     
      if show_inference                                                    
          ruleview(fis);                                                % rule window opening
          xx = allchild(0);                                             % all child windows
          for i=1:length(xx)                                            
              name=get(xx(i), 'name');
              if length(name)>11 && strcmp(name(1:12), 'Rule Viewer:');
                  figReg=xx(i);                                         
                  break;
              end
          end
          ekran = get(0,'screensize');                                  % screen size
          %set(figN, 'Position',[ekran(3)/2,0,ekran(3)/2,ekran(4)/2]);
          set(figReg, 'HandleVis', 'on');                               
          pause
      end
      
      for k=1:step,
         y = ykab+history(k,2);
         set(idkab,'xdata', xkab, 'ydata', y);
         yciag = [-history(k,4)*max_x/1000 0 0 -history(k,4)*max_x/1000] + history(k,2);
         set(idciag,'xdata', xciag, 'ydata', yciag);
         %set(idl,'xdata', xs, 'ydata', ys);
         drawnow;
         
         if show_inference
             ruleview('#simulink', history(k,1:end-1), figReg);   
         end
         
         pause(0.01)
      end
      %delete(idkab);
      close;
   end
   
end
% Displaing results:
disp(sprintf(' <Number of init.state>:<landing speed>  <final amount of fuel> <number of steps>'));
for i=1:number_of_initial_states  
   disp(sprintf(' %d: %f     %f     %d',i,V_final(i),fuel_final(i),last_step(i))); 
end
