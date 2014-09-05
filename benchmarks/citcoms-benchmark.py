'''
citcoms-benchmark.py
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

d=0.45
kappa=1.0
time_rescale = (d*d)/kappa # t_citcom / time_rescale = t_nondim (or t_paper) 
vel_rescale = kappa / d    # v_citcom / vel_rescale = v_nondim (i.e. v_paper)
r_t = 1.0
r_b = 0.55
top_prefac = r_t*(r_t-r_b)/r_b;

dataA1 = np.loadtxt("/home/rkk/A1_10/A1_10.volume_avg")
dataA3 = np.loadtxt("/home/rkk/A3_08/A3_08.volume_avg")
dataA8 = np.loadtxt("/home/rkk/A8_05/A8_05.volume_avg")
surfA1 = top_prefac*np.loadtxt("/home/rkk/A1_10/surface-heat-flux-A1_10")
surfA3 = top_prefac*np.loadtxt("/home/rkk/A3_08/surface-heat-flux-A3_08")
surfA8 = top_prefac*np.loadtxt("/home/rkk/A8_05/surface-heat-flux-A8_05")
timeA1 = (1.0/time_rescale)*dataA1[:,1]
timeA3 = (1.0/time_rescale)*dataA3[:,1]
timeA8 = (1.0/time_rescale)*dataA8[:,1]
T_avgA1 = dataA1[:,2]
T_avgA3 = dataA3[:,2]
T_avgA8 = dataA8[:,2]
Vrms_avgA1 = (1.0/vel_rescale)*dataA1[:,3]
Vrms_avgA3 = (1.0/vel_rescale)*dataA3[:,3]
Vrms_avgA8 = (1.0/vel_rescale)*dataA8[:,3]

plt.figure(1)
# plt.subplot(3,1,1)
plt.plot(timeA1,T_avgA1, 'g-', label='A1')
plt.plot(timeA3,T_avgA3, 'r-', label='A3')
plt.plot(timeA8,T_avgA8, 'b-', label='A8')
plt.legend(title='Benchmark', loc='best')
plt.xlabel('time')
plt.ylabel('<T>')
plt.title('Time dependence of Volume-averaged temperature')
plt.ylim((0,0.8))
plt.xlim((0,1.0))
plt.yticks([0.0,0.2,0.4,0.6,0.8])

plt.figure(2)
# plt.subplot(3,1,2)
plt.plot(timeA1,Vrms_avgA1, 'g-', label='A1')
plt.plot(timeA3,Vrms_avgA3, 'r-', label='A3')
plt.plot(timeA8,Vrms_avgA8, 'b-', label='A8')
plt.legend(title='Benchmark', loc='best')
plt.xlabel('time')
plt.ylabel('<V_rms>')
plt.title('Time dependence of Root-mean squared velocity')
plt.ylim((0,120.0))
plt.xlim((0,1.0))
plt.yticks(np.arange(0.0,121.0,20.0))

plt.figure(3)
# plt.subplot(3,1,3)
plt.plot(timeA1,surfA1, 'g-', label='A1')
plt.plot(timeA3,surfA3, 'r-', label='A3')
plt.plot(timeA8,surfA8, 'b-', label='A8')
plt.legend(title='Benchmark', loc='best')
plt.xlabel('time')
plt.ylabel('Nu_t')
plt.title('Time dependence of surface Nusselt number')
plt.ylim((0,6.0))
plt.xlim((0,1.0))

plt.show()
