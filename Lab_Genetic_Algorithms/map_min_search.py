"""
 2-D function drawing with points

"""
import numpy as np
#import math
import matplotlib.pyplot as plt

def fun3(x1,x2):
    eps = 0.00000001
    R1 = np.sqrt(0.3*(x1+3)**2 + (x2+4)**2 + eps)
    R2 = np.sqrt(0.2*(x1-7)**2 + (x2-6)**2 + eps)
    R3 = np.sqrt(0.2*(x1-7)**2 + 0.5*(x2-6)**2 + eps)
    R4 = np.sqrt(0.7*(x1+7)**2 + 2*(x2-6)**2 + eps)
    R5 = np.sqrt(0.2*(x1+3)**2 + 0.05*(x2+5)**4 + eps)

    y = np.sin(x1*3)/(abs(x1)+1) + np.sin(x2*5-1)/(abs(x2/2-1)+1) + ((x1-5)**2+(x2-5)**2)/50 + \
        4*np.sin(R1)/R1 + 4*np.sin(R2)/R2 - 3*np.sin(R4)/R4 - 3*np.sin(R5)/R5
    return y

def draw_map():
    X1, X2 = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
    Z = fun3(X1,X2)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    cs = plt.contour(X1,X2,Z, levels=[-2.0, -1.8, -1.5, -1, -0.5, 0, 1, 2, 3, 4, 5,7,10])
    #fig,ax = plt.subplots()
    #CS = ax.contour(X1,X2,Z, levels=[-2.0, -1.8, -1.5, -1, -0.5, 0, 1, 2, 3, 4, 5,7,10])
    #ax.clabel(CS, inline=True, fontsize=10)

def show_fun():

    draw_map()
    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    plt.plot(x, y, 'o', color='black');
    plt.title('2D map: highlands - yellow, valleys - blue')
    plt.show()

def show_population(P,title):
    draw_map()
    plt.plot(P[0,:], P[1,:], 'o', color='red');
    plt.title(title)
    plt.show()

def show_the_best(x,title):
    draw_map()
    plt.plot(x[0], x[1], '+', color='red');
    plt.title(title)
    plt.show()


#show_fun()


