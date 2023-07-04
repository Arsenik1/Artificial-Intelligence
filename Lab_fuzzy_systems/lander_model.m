% funkcja obliczajaca reakcje ladownika na pobudzenie
% Pobudzenie: sila F, ilosc paliwa mp, odleglosc x, predkosc v.
% Odpowiedz: odleglosc x, predkosc v, ilosc paliwa mp.

function state_next=lander_model(state,F)

global g t ml k ko Fmax

mp=state(1);
x=state(2);
v=state(3);

mc=ml+mp; % own mass + fuel mass

if F<0
   F=0;
end
if F>Fmax
   F=Fmax;
end

mp1=mp-k*F*t;
if mp1<0
    F=mp/(k*t);
    mp1=0;
end
%if mp == 0  
%   F = 0
%end   




if F>0.1
    v1=ko*log(mc/(mc-k*F*t))+v-g*t;
    x1=-(ko*ko/F)*log(mc/(mc-k*F*t))*(mc-k*F*t)+t/k+v*t-(g*t*t)/2+x;
else
    v1=v-g*t;
    x1=v*t-(g*t*t)/2+x;
end

% determination of the exact moment of impact and speed at the moment.

if x1<0
    x1=0;
    tpom=0:0.01:t;
    i=1;
    if F>0.1    
        while -(ko*ko/F)*log(mc/(mc-k*F*tpom(i)))*(mc-k*F*tpom(i))+tpom(i)/k+v*tpom(i)-(g*tpom(i)*tpom(i))/2+x>0,
            i=i+1;
        end
        v1=ko*log(mc/(mc-k*F*tpom(i)))+v-g*tpom(i);
    else
        while v*tpom(i)-(g*tpom(i)*tpom(i))/2+x>0,
            i=i+1;
        end
        v1=v-g*tpom(i);        
    end
end

state_next=[mp1,x1,v1];
