#-----------------------------------------------------------------------------------------------------------------------
# table6.py : Generate the entries for Table 6 of the Zhong et. al. paper
#-----------------------------------------------------------------------------------------------------------------------
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

def get_array_slice(arr, low, high):
  '''
  returns a boolean array with True for arr in[low,high]
  '''

  lower_limit = np.greater_equal(arr, low)
  upper_limit = np.less_equal(arr, high)
  combined = np.logical_and(lower_limit, upper_limit)
  return combined

def compute_and_print_mean_and_std(benchmark, volume_avg_file, heat_flux_file, tlow, thigh):
  data = np.loadtxt(volume_avg_file)
  surf = top_prefac * np.loadtxt(heat_flux_file)
  time = (1.0/time_rescale) * data[:,1]
  T_avg = data[:,2]
  Vrms_avg = (1.0/vel_rescale)*data[:,3]
  combined = get_array_slice(time, tlow, thigh)
  sub_Tavg = T_avg[combined]
  sub_Vrms = Vrms_avg[combined]
  sub_Nu_t = surf[combined]

  print("----------------- %s Benchmark Results -----------------------" % benchmark)
  print("<T> = %8.4f std = %e" % (np.mean(sub_Tavg), np.std(sub_Tavg)))
  print("<V_rms> = %8.4f std = %e" % (np.mean(sub_Vrms), np.std(sub_Vrms)))
  print("<Nu_t> = %8.4f std = %e" % (np.mean(sub_Nu_t), np.std(sub_Nu_t)))
  print("")

compute_and_print_mean_and_std("A1", 
  "/home/rkk/A1_10/A1_10.volume_avg", 
  "/home/rkk/A1_10/surface-heat-flux-A1_10", 
  0.7, 1.0)

'''
compute_and_print_mean_and_std("A2", 
  "/home/rkk/A2_02/A2_02.volume_avg", 
  "/home/rkk/A2_02/surface-heat-flux-A2_02", 
  1.0, 1.3)
'''
compute_and_print_mean_and_std("A3", 
  "/home/rkk/A3_08/A3_08.volume_avg", 
  "/home/rkk/A3_08/surface-heat-flux-A3_08", 
  0.6, 0.9)
'''
compute_and_print_mean_and_std("A4", 
  "/home/rkk/A4_02/A4_02.volume_avg", 
  "/home/rkk/A4_02/surface-heat-flux-A4_02", 
  1.5, 2.0)

compute_and_print_mean_and_std("A5", 
  "/home/rkk/A5_01/A5_01.volume_avg", 
  "/home/rkk/A5_01/surface-heat-flux-A5_01", 
  1.0, 1.5)

compute_and_print_mean_and_std("A7", 
  "/home/rkk/A7_01/A7_01.volume_avg", 
  "/home/rkk/A7_01/surface-heat-flux-A7_01", 
  1.2, 1.7)
'''
compute_and_print_mean_and_std("A8", 
  "/home/rkk/A8_05/A8_05.volume_avg", 
  "/home/rkk/A8_05/surface-heat-flux-A8_05", 
  0.8, 1.0)
'''
compute_and_print_mean_and_std("B1", 
  "/home/rkk/B1_01/B1_01.volume_avg", 
  "/home/rkk/B1_01/surface-heat-flux-B1_01", 
  1.2, 1.7)

compute_and_print_mean_and_std("B2", 
  "/home/rkk/B2_01/B2_01.volume_avg", 
  "/home/rkk/B2_01/surface-heat-flux-B2_01", 
  0.8, 1.1)

compute_and_print_mean_and_std("B3", 
  "/home/rkk/B3_01/B3_01.volume_avg", 
  "/home/rkk/B3_01/surface-heat-flux-B3_01", 
  0.8, 1.2)

compute_and_print_mean_and_std("B4", 
  "/home/rkk/B4_01/B4_01.volume_avg", 
  "/home/rkk/B4_01/surface-heat-flux-B4_01", 
  1.0, 1.3)
'''

