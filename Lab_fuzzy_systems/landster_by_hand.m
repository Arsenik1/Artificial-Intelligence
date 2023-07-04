% SYMULACJA LADOWNIKA - STEROWANIE RECZNE 
% sila ciagu zalezy od polozenia wskaznika myszki w okienku z prawej strony
% zmiana skali za pomoca klawiszy '-' - pomniejszenie, '+' - powiekszenie
% dane zapisywane sa po kliknieciu klawisza 'z' przy aktywnym oknie symulacji 
% do pliku historia.txt. Moga posluzyc do uczenia w modelu Sugeno: najpierw
% nalezy wygenerowac szablon systemu rozmytego np. fis = genfis1(...), a
% nastepnie uruchomic alg. uczenia: fis2 = anfis(fis,...)

clear
land_glob;  % zmienne globalne
global Fmax;

% PARAMETRY DOBIERANE INNDYWIDUALNIE:
opoznienie_czasowe = 0.2;              % odstep czasowy pomiedzy krokami
dH = 500;                              % wysokosc fragmentu przestrzeni pokazywanej na ekranie
stany_pocz = [ 200 220 0;1000 3000 10; 400 2000 -60;800 4000 -100; 400 50 150];  % zbior stanow poczatkowych

                                     % kazdy stan opisany przez trojke [ilosc_paliwa, x, V]
liczba_stanow_pocz = length(stany_pocz(:,1))                         

%fis = readfis('fis_max.fis');                            % odczyt systemu rozmytego z pliku
liczba_krokow = 5000;
V_konc = [];
ilosc_konc = [];
krok_konc = [];
historia = [];

disp(sprintf('Reczne starowanie ladownikiem - gromadzenie danych do uczenia np. w systemie anfis '));
disp(sprintf('Dane w postaci wierszy 4-elementowych: <parametry stanu> <sila> zapisywane sa w pliku historia.txt '));
disp(sprintf('wcisnij klawisz...'));
pause

for st=1:liczba_stanow_pocz                      % po stanach poczatkowych
   stan = stany_pocz(st,:);
   x = stan(2);
   krok = 0;
   historia = [];   
   F=0;
   
   gca1=subplot(1,2,1);
   gca2=subplot(1,2,2);
   set(gca1,'Position',[0.10 0.05 0.55 0.85]);
   PozycjaSily = [0.80 0.05 0.15 0.85];         % x_pocz, y_pocz, szer, wys okna do wskazywania sily (okno gca2)
   set(gca2,'Position',PozycjaSily);
   
   max_x = x;
   
   %subplot(gca1)
   numer_ax = ceil(x/dH);
   axis(gca1,[-40  40  (numer_ax-1)*dH-dH/8  (numer_ax+0.2)*dH]);

   %subplot(gca2)
   axis(gca2,[0  5  -Fmax  0]);
   grid on;
   ylabel('duza <-----  sila  -----> mala');
   subplot(gca1);
  
   ylabel('wysokosc [m]');

   xkab = [-2 -1 -4 -3 3 4 1 2];
   ykab = [0 2 2 6 6 2 2 0]*dH/100;
   xciag = [-2 -2 2 2];
   yciag = 0;
   y = ykab + x;
   idkab = patch(xkab,y , 'b', 'erasemode','xor');     % rysowanie wypelnionego konturu kadluba
   idciag = patch(xciag,yciag, 'y','erasemode','xor');  % rysowanie plomienia ciagu
   idgrunt = patch([-40 -40 40 40],[-dH/8 0 0 -dH/8] , 'g', 'erasemode','xor');   % rys. powierzchni planety
   idpaliwo = patch([-38 -38 -28 -28],[dH*(numer_ax-1)   dH*(numer_ax-1)+stan(1)/1000*dH/4 ...
                                       dH*(numer_ax-1)+stan(1)/1000*dH/4   dH*(numer_ax-1)],'r', 'erasemode','xor'); 
   figure(gcf);
   disp(sprintf('stan poczatkowy:  mp = %f   x = %f   V = %f',stan(1),stan(2),stan(3)));
   disp(sprintf('ustaw wskaznik myszy u gory okna po prawej stronie, a nastepnie wcisnik klawisz...'));
   pause
   
   while (krok < liczba_krokow)&(x~=0)     % po krokach czasowych az do wyladowania   
      ilosc_paliwa = stan(1);
      x = stan(2);
      V = stan(3);
      
      %F = evalfis([ilosc_paliwa x V],fis); % wyznaczenie sily ciagu za pomoca systemu rozmytego
      
      % Wyznaczenie sily za pomoca wskaznika myszy w oknie gca2:
      Po = get(gcf, 'Position');              % pozycja biezacego okna na ekranie
      Pm = get(0,'PointerLocation');          % polozenie wskaznika myszy na calym ekranie
      %Sc = get(0,'ScreenSize');               % rozmiary ekranu
      % wspolrzedne wskaznika myszy jak proporcje wewnatrz okna figure:
      a = (Pm(1) - Po(1))/Po(3);
      b = (Pm(2) - Po(2))/Po(4);

      if (a > PozycjaSily(1))&(a < PozycjaSily(1)+PozycjaSily(3))
          F = (1 - (b-PozycjaSily(2))/PozycjaSily(4))*Fmax;
          if F<0
              F=0;
          end
          if F>Fmax
              F=Fmax;
          end
      else 
          F = 0;
      end
      
      if ilosc_paliwa == 0
          F = 0;
      end
      
      % Rysowanie ciagu pokazanego wskaznikiem w oknie gca2:
      %subplot(gca2);
      %yciag = [-F/100*max_x/1000 0 0 -F/100*max_x/1000] + x;
      %set(idciag,'xdata', xciag, 'ydata', yciag);
      
      subplot(gca1);
      Ax = axis(gca1);   
      
      ch = get(gcf, 'CurrentCharacter');
      % Zmiana skali 
      if ch=='-'
          dH = 2*dH;
          set(gcf,'CurrentCharacter','~');
          figure(gcf);
      elseif (ch=='=')|(ch=='+')
          dH = dH/2;
          set(gcf,'CurrentCharacter','~');
          figure(gcf);
      end
      
      % Zmiana obszaru przestrzeni, gdy ladownik poleci zbyt wysoko lub
      % nisko:
      numer_ax = ceil(x/dH);     % numer obszaru
      if numer_ax < 1 
          numer_ax=1;
      end
      if Ax(4) ~= numer_ax*dH
          axis(gca1,[-40  40  (numer_ax-1)*dH-dH/8  (numer_ax+0.2)*dH]);
      end
      
      
      
      ykab = [0 2 2 6 6 2 2 0]*dH/100;
      y = ykab + x;
      set(idkab,'xdata', xkab, 'ydata', y);
      yciag = [-F/100*dH/1000 0 0 -F/100*dH/1000] + x;
      set(idciag,'xdata', xciag, 'ydata', yciag);
      set(idpaliwo,'ydata', [dH*(numer_ax-1)   dH*(numer_ax-1)+ilosc_paliwa/1000*dH/4 ...
                             dH*(numer_ax-1)+ilosc_paliwa/1000*dH/4   dH*(numer_ax-1)]);
      set(idgrunt,'ydata',[-dH/8 0 0 -dH/8]);               

      drawnow;
      pause(opoznienie_czasowe);
      
      stan_n = lander_model(stan,F);           % wyznaczenie nowego stanu po pewnym interwale czasowym
      stan = stan_n;
      krok = krok+1;
            
      historia(krok,:) = [stan F];         % zapis biezacego stanu + proponowanej sily ciagu (nie musi taka byc)
   end  % po krokach w danym stanie
   lan = sprintf('stan koncowy:  mp_k = %f   V_k = %f', ilosc_paliwa,V);
   disp(lan);
   subplot(gca1);
   text(-32,dH/4,lan);
   disp(sprintf('wcisnij z, zeby zapisac w pliku lub inny klawisz zeby przejsc dalej...'));
   ch = '~';set(gcf,'CurrentCharacter','~')
   while (ch == '~')
       ch = get(gcf, 'CurrentCharacter');
       pause(0.1);
   end   
   
   if (ch=='z')
       f = fopen('historia.txt','a');
       fprintf(f,'\n%s mp = %f   x = %f   V = %f\n','% stan poczatkowy:',stany_pocz(st,1),stany_pocz(st,2),stany_pocz(st,3));  
       for i=1:krok
          fprintf(f,'%s\n',num2str(historia(i,:)));
       end   
       fclose(f);
   end
  
   
   V_konc(st) = V;                          % zapis wartosci koncowych
   ilosc_konc(st) = ilosc_paliwa;
   krok_konc(st) = krok;   
end   % po stanach poczatkowych
close;

% Wyswietlanie wynikow:
disp(sprintf(' <Nr stanu pocz.>:<predkosc ladowania>  <ilosc paliwa> <liczba krokow>'));
for i=1:liczba_stanow_pocz  
   disp(sprintf(' %d: %f     %f     %d',i,V_konc(i),ilosc_konc(i),krok_konc(i))); 
end

punktacja = 10000/sum(abs(V_konc.^4)) + sum(ilosc_paliwa)*(max(abs(V_konc))<5)