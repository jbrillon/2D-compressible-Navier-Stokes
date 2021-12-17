# Import libraries
import numpy as np # NumPy: contains basic numerical routines
import scipy # SciPy: contains additional numerical routines to numpy
import matplotlib.pyplot as plt # Matlab-like plotting
import scipy.sparse as scysparse
import scipy.sparse.linalg

data_fileType = 'txt'
terms = ['first_derivatives','euler']
term_names = [r'ddx',r'ddy']
components = ['x','y']
subdirectories = ['Data/','Figures/OoA/']

eqn = 'energy'
term_index = 1
terms_avail = np.array([term_index],dtype=int)
components_avail = np.array([0,1],dtype=int)

print('=====================================================')
for j in components_avail:
	figure_title = "Root-Mean-Square of Error vs Inverse of Spatial Grid Spacing"
	figure_title_print = "Root-Mean-Square of Error vs Inverse of Spatial Grid Spacing"
	print('Plotting: ' + figure_title_print)
	fig = plt.figure(figure_title)
	plt.grid()
	plt.title(figure_title,fontsize=12)#,fontweight='bold')
	plt.xlabel(r'$h^{-1} = (\sqrt{\Delta x\Delta y})^{-1}$',fontsize=12)
	plt.ylabel(r"$\epsilon^{RMS}_{TR}$",rotation=90,fontsize=12)

	for i in terms_avail:
		# dx
		subdirectory = subdirectories[0]
		filename = "dx_store_%s_%s_%s" % (terms[i],components[j],eqn)
		dx_store = np.loadtxt(subdirectory+filename+'.'+data_fileType,unpack=False)
		# RMS error
		filename = "RMS_error_dx_%s_%s_%s" % (terms[i],components[j],eqn)
		RMS_store_dx = np.loadtxt(subdirectory+filename+'.'+data_fileType,unpack=False)
		
		# plt.xlim([1e0,1e3])
		# max_val_current = np.max(RMS_store_dx)
		# max_val = np.min([max_val_current,np.max(RMS_store_dx)])
		# min_val_current = np.min(RMS_store_dx)
		# min_val = np.min([min_val_current,np.min(RMS_store_dx)])
		# plt.ylim([min_val,max_val])
		
		# testing
		plt.xlim([1e0,1e2])
		# plt.ylim([1e-3,1e0])

		name = term_names[j]
		plt.loglog(1.0/dx_store,RMS_store_dx,label=name)
	
	# Confirm the order of accuracy
	for n in range(1,4):
		if j==0:
			name = r"$\Delta x^{-%i}$" % n
		elif j==1:
			name = r"$\Delta y^{-%i}$" % n

		shift = -4.8

		dx_n = (dx_store**n)*10.0**(shift+float(n))
		plt.loglog(1.0/dx_store,dx_n,label=name,linewidth=0.5,linestyle='--')

	plt.tight_layout()
	leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True, fontsize=8)
	print(' ... Saving figure ...')
	if j==0:
		figure_name = "RMS_error_" + terms[i] + "_x_" + eqn
	elif j==1:
		figure_name = "RMS_error_" + terms[i] + "_y_" + eqn
	figure_fileType = 'png'
	subdirectory = subdirectories[1]
	plt.savefig(subdirectory + figure_name + '.' + figure_fileType,format=figure_fileType,dpi=500)
	plt.close()
	
	# print('-----------------------------------------------------')
	
	# figure_title = "Root-Mean-Square of Error vs Inverse of Temporal Grid Spacing\n for %s Strategies" % (temp_discs[i])
	# figure_title_print = "Root-Mean-Square of Error vs Inverse of Temporal Grid Spacing for %s Strategies" % (temp_discs[i])
	# print('Plotting: ' + figure_title_print)
	# fig = plt.figure(figure_title)
	# plt.grid()
	# plt.title(figure_title,fontsize=12,fontweight='bold')
	# plt.xlabel(r'$\Delta t^{-1}$',fontsize=12)
	# plt.ylabel(r"$\epsilon^{RMS}_{TR}$",rotation=90,fontsize=12)

	# for j in lin_adv_disc_avail:
	# 	# dx
	# 	subdirectory = subdirectories[0]
	# 	filename = "dt_store_method_%i_%i" % (i,j)
	# 	dt_store = np.loadtxt(subdirectory+filename+'.'+data_fileType,unpack=False)
	# 	# RMS error
	# 	filename = "RMS_error_dt_method_%i_%i" % (i,j)
	# 	RMS_store_dt = np.loadtxt(subdirectory+filename+'.'+data_fileType,unpack=False)
		
	# 	# plt.xlim([1e0,1e3])
	# 	plt.ylim([1e-3,1e0])
	# 	name = temp_discs[i] + ' with ' + lin_adv_discs[j]
	# 	plt.loglog(1.0/dt_store,RMS_store_dt,label=name)
	
	# # Confirm the order of accuracy
	# n_time = [1,2,3]
	# for n in n_time: 
	# 	name = r"$\Delta t^{-%i}$" % n
	# 	if n == 2:
	# 		shift = 2.0
	# 	else:
	# 		shift = 0.0
	# 	if n == 3:
	# 		shift = 3.0
	# 	dt_n = (dt_store**n)*10.0**(shift+float(n))
	# 	plt.loglog(1.0/dt_store,dt_n,label=name,linewidth=0.5,linestyle='--')

	# plt.tight_layout()
	# leg = plt.legend(loc='best', ncol=2, shadow=True, fancybox=True, fontsize=8)
	# print(' ... Saving figure ...')
	# figure_name = "RMS_error_dt"
	# figure_fileType = 'eps'
	# subdirectory = subdirectories[1]
	# plt.savefig(subdirectory + figure_name + '.' + figure_fileType,format=figure_fileType,dpi=500)
	# plt.close()
	# print('-----------------------------------------------------')
print('=====================================================')





