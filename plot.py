import matplotlib.pyplot as plt
import numpy as np

BXT_dat=np.genfromtxt('Data/BXT_curve.csv')
ADA_dat=np.genfromtxt('Data/ADA_curve.csv')
RF_dat=np.genfromtxt('Data/RF_curve.csv')
DNN_dat=np.genfromtxt('Data/DNN_curve.csv')

models={
    'BXT':BXT_dat,
    'RF':RF_dat,
    'ADA':ADA_dat,
    'DNN':DNN_dat       
       }

for mtype,m in models.items():
    plt.figure()
    plt.plot(m[:,0],m[:,1])
    plt.xlabel('Decision Threshold')
    plt.ylabel('AMS')
    plt.title(f'{mtype}_AMS vs Decision Threshold')
    plt.savefig(f'Plots/{mtype}_ams_plot.svg')



plt.figure()
plt.plot(BXT_dat[:,0],BXT_dat[:,1],label='BXT')
plt.plot(RF_dat[:,0],RF_dat[:,1],label='Random Forest')
plt.plot(DNN_dat[:,0],DNN_dat[:,1],label='DNN')
plt.plot(ADA_dat[:,0],ADA_dat[:,1],label='BXT')
plt.xlabel('Decision Threshold')
plt.ylabel('AMS')
plt.title('AMS vs Decision Threshold')
plt.legend()
plt.savefig('Plots/final_ams_plot.svg')