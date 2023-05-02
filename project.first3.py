#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:59:34 2023

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
n1=np.random.normal(mu,sigma,N)
n2=np.random.normal(mu,sigma,N)

v0=0
r0=0
#def f(v,dt):
 #   dv=-y*v+((A*n)/dt)*math.sqrt(dt)
  #  return dv

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


t=0
T=10000
v2squared=vsquared[t:t+T]
averagevsquared=np.mean(v2squared)
print(averagevsquared)

rsquared=np.zeros((N))
for i in range(N-1):
    rsquared[i]=(r[i,0])**2+(r[i,1])**2


t1=1000
T1=1000
r1squared=rsquared[t1:t1+T1]
averager1squared=np.mean(r1squared)
print(averager1squared)


t2=0
T2=9999+1
r2squared=rsquared[t2:t2+T2]
averager2squared=np.mean(r2squared)
print(averager2squared)


t3=2000
T3=4000
r3squared=rsquared[t3:t3+T3]
averager3squared=np.mean(r3squared)
print(averager3squared)


t4=5000
T4=1000
r4squared=rsquared[t4:t4+T4]
averager4squared=np.mean(r4squared)
print(averager4squared)

print(len(r4squared))

plt.plot(v[:,0],v[:,1])
plt.plot(r[:,0],r[:,1])
plt.xlim(-500,+500)
plt.ylim(-500,+500)
plt.title("i dunno")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()