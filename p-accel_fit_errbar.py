import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv

mpl.rcParams['font.family'] = 'monospace'
mpl.rcParams['axes.linewidth'] = 1.8
mpl.rcParams['xtick.major.width']=1.3
mpl.rcParams['ytick.major.width']=1.3

SMALL_SIZE = 14
MEDIUM_SIZE = 17
BIGGER_SIZE = 20
#HOLA
#Definition of the imput parameters

tab = pd.read_csv('NGC6440H_data_accelsearch.txt', header = None, delimiter = " ")
c = 2.99792458e8

Obs = tab[0]
P = np.array(tab[1])
err_p = np.array(tab[2])
accel = np.array(tab[3])
err_accel = np.array(tab[4])

#Definition of the Ellipse

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

#Fit including the error bars and the centre 

scale = 1e4
R = np.linspace(0, 2*np.pi, 100)
x = (P-2.8484)*scale
n = 100000
y2 = np.random.normal(accel,err_accel,(n,len(P)))
x2 = np.random.normal(x,err_p,(n,len(P)))

xc, yc, a, b = [], [], [], []

for i in range(n) :
    A = fitEllipse(x2[i],y2[i])
    center = ellipse_center(A)
    phi = 0
    axes = ellipse_axis_length(A)
    xc.append(center[0])
    yc.append(center[1])
    a.append(np.min(axes))
    b.append(np.max(axes))

#Get rid of the 'nan'

a = np.array(a)
b = np.array(b)
xc = np.array(xc)
yc = np.array(yc)
ind = np.where(~np.isnan(a))[0]    

a0, b0 = np.mean(a[ind]), np.mean(b[ind])
center = np.mean(xc[ind]), np.mean(yc[ind])

center = list(center)

#Fitting the ellipse and errors

def ellipse(center, a0, b0) :
    x = np.real(center[0] + a0*np.cos(R)*np.cos(phi) - b0*np.sin(R)*np.sin(phi))
    y = np.real(center[1] + a0*np.cos(R)*np.sin(phi) + b0*np.sin(R)*np.cos(phi))
    return x, y

xx, yy = ellipse(center, a0, b0)
xx = xx*1/scale

astd = np.std(a[ind]) 
bstd = np.std(b[ind]) 
xxmax, yymax = ellipse(center, a0+astd, b0+bstd)
xxmin, yymin = ellipse(center, a0-astd, b0-bstd)

xxmax = xxmax*1/scale
xxmin = xxmin*1/scale

#Obtaining the results

print("center = ",  center)
print("angle of rotation = ",  phi)
print("axes = ", a0*1/scale,b0) 

center[0] = center[0]*1/scale 

x0 = center[0]+2.8484
a0 = a0*1/scale

pb = ((a0*1e-3)*2*np.pi*c)/((x0*1e-3)*b0)
a1 = ((a0*1e-3)/(x0*1e-3))**2*(c/b0)

print("The orbital period is", pb/60/60/24)
print("The projected semi-major axis", a1)

xerr = np.std(xc)*1/scale
yerr = np.std(yc)
print("The errors of the fit parameters are: \nThe errors of the center (x,y):", xerr,yerr)
print("The errors of the axis of the ellipse (semi-major, semi-minor):", astd*1/scale, bstd)

#Plotting

plt.xlabel('Period + 2.8484 [ms]', fontsize = SMALL_SIZE)
plt.ylabel('Acceleration [ms$^{-2}$]', fontsize = SMALL_SIZE)
plt.title('NGC 6440H', fontsize = SMALL_SIZE)

from pylab import *

plot(xx,yy, color = 'dodgerblue', lw = 2)
plot(xxmax, yymax, color = 'gray')
plot(xxmin, yymin, color = 'gray')
errorbar(x*1/scale, accel, xerr = err_p, yerr = err_accel, ecolor='black', fmt='ro')
errorbar(*center, xerr = np.std(xc)*1/scale, yerr = np.std(yc), ecolor='gray', fmt='bP')
show()
