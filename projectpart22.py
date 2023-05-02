#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:07:58 2023

@author: spurcell
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

tmin=0
tmax=1000
dt=0.1
y=0.2
A=4
mu=0
sigma=1
t=np.arange(tmin,tmax+dt,dt)
N=len(t)
N=len(t)
v0=0
r0=0

def Rfunc():


    n1=np.random.normal(mu,sigma,N)
    n2=np.random.normal(mu,sigma,N)


    v = np.zeros((N,2))
    v[0,:]=v0

    for i in range(N-1):
        v[i+1,0] = v[i,0] + (-y*v[i,0])*dt+(A*n1[i+1])*math.sqrt(dt)
        v[i+1,1] = v[i,1] + (-y*v[i,1])*dt+(A*n2[i+1])*math.sqrt(dt)

    r = np.zeros((N,2))
    r[0,:]=0

    for n in range(N-1):
        r[n+1,0]= r[n,0]+dt*v[n,0]
        r[n+1,1]= r[n,1]+dt*v[n,1]

    vsquared=np.zeros((N))
    for i in range(N-1):
        vsquared[i]=v[i,0]**2+v[i,1]**2

    rsquared=np.zeros((N))
    for i in range(N-1):
        rsquared[i]=(r[i,0])**2+(r[i,1])**2


    return rsquared

def Vfunc():

    n1=np.random.normal(mu,sigma,N)
    n2=np.random.normal(mu,sigma,N)


    v = np.zeros((N,2))
    v[0,:]=v0

    for i in range(N-1):
        v[i+1,0] = v[i,0] + (-y*v[i,0])*dt+(A*n1[i+1])*math.sqrt(dt)
        v[i+1,1] = v[i,1] + (-y*v[i,1])*dt+(A*n2[i+1])*math.sqrt(dt)

    r = np.zeros((N,2))
    r[0,:]=0

    for n in range(N-1):
        r[n+1,0]= r[n,0]+dt*v[n,0]
        r[n+1,1]= r[n,1]+dt*v[n,1]

    vsquared=np.zeros((N))
    for i in range(N-1):
        vsquared[i]=v[i,0]**2+v[i,1]**2

    rsquared=np.zeros((N))
    for i in range(N-1):
        rsquared[i]=(r[i,0])**2+(r[i,1])**2


    return vsquared



particleR=np.zeros((N,100))

for j in range(100):
    particleR[:,j]=Rfunc()









print(particleR[1000,0])
print(particleR[1000,1])
print(particleR[1000,98])



Rsquared=np.zeros(N)

for i in range(N):
    Rsquared[i]=np.mean(particleR[i,:])


print(Rsquared[0])
print(Rsquared[1000])
print(Rsquared[7345])
print(Rsquared[-1])



particleV=np.zeros((N,100))

for j in range(100):
    particleV[:,j]=Vfunc()

Vsquared=np.zeros(N)

for i in range(N):
    Vsquared[i]=np.mean(particleV[i,:])

print(Vsquared[0])
print(Vsquared[1000])
print(Vsquared[7345])
print(Vsquared[-1])
print(Vsquared[-2])


tmin=0
tmax=1000
dt=0.1
y=0.2
A=4
mu=0
sigma=1
t=np.arange(tmin,tmax+dt,dt)
N=len(t)
n1=np.random.normal(mu,sigma,N)
n2=np.random.normal(mu,sigma,N)

v0=0
r0=0


N=len(t)
v = np.zeros((N,2))
v[0,:]=v0

for i in range(N-1):
    v[i+1,0] = v[i,0] + (-y*v[i,0])*dt+(A*n1[i+1])*math.sqrt(dt)
    v[i+1,1] = v[i,1] + (-y*v[i,1])*dt+(A*n2[i+1])*math.sqrt(dt)

r = np.zeros((N,2))
r[0,:]=0

for n in range(N-1):
    r[n+1,0]= r[n,0]+dt*v[n,0]
    r[n+1,1]= r[n,1]+dt*v[n,1]




vsquared=np.zeros((N))
for i in range(N-1):
    vsquared[i]=v[i,0]**2+v[i,1]**2




rsquared=np.zeros((N))
for i in range(N-1):
    rsquared[i]=(r[i,0])**2+(r[i,1])**2


risquared=np.zeros(N)
averagersquared=np.zeros(N)

#t0=0
#risquared[i]=rsquared[t0:t0+i]
#averagersquared[i]=np.mean(risquared[i])
#print(averagersquared[0])

plt.plot(t[0:-1],Vsquared[0:-1])
#plt.plot(r[:,0],r[:,1])
#plt.xlim(-10000,+10000)
#plt.ylim(-10000,+10000)
plt.title("i dunno")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()



#quared=np.array(rsquared)
#for i in range(N):
 #      for j in range(100):
  #        n1=np.random.normal(mu,sigma,N)
   #        n2=np.random.normal(mu,sigma,N)
    #     Rsquared[i]=np.append(Rsquared[i],rsquared[i])

 #  P=np.zeros(N)
#   for i in range(N):
 #      P[i]=np.mean(Rsquared[i])