#=====================================================
# Generates colourful spy plots of matrix patterns
#=====================================================
# Import libraries
import numpy as np # NumPy: contains basic numerical routines
import scipy # SciPy: contains additional numerical routines to numpy
import matplotlib.pyplot as plt # Matlab-like plotting
import scipy.sparse as scysparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.colors as colors
#=====================================================
data_fileType = 'txt'
subdirectories = ['Data/','Figures/SPY/']
#=====================================================
# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
#=====================================================
# Load matrix A
BCs = ['Periodic','ChannelFlow'] #['Dirichlet','Neumann']
operators = ['L','ddx','ddy','ddx_Sxx','ddy_Syy','ddy_Sxy_u','ddy_Sxy_v','ddx_Sxy_u','ddx_Sxy_v']
operator_name = ['Laplacian', 'First Derivative w.r.t. x', 'First Derivative w.r.t. y','','','','','','']

# INPUTS
eqn = 'ghostCell'
i = 2 # select operator (index)

if i == 2:
	matrix_shape = 'rectangular'
	BC = BCs[1]
elif i == 1:
	matrix_shape = 'square'
	BC = BCs[0]

if i < 3:
	if matrix_shape == 'rectangular' and eqn == 'ymomentum':
		Nxc = 4
		Nyc = 4
	else:
		if eqn == 'ymomentum':
			Nxc = 4
			Nyc = 3
		elif eqn == 'ghostCell' or eqn == 'boundary':
			Nxc = 4
			Nyc = 5
		else:
			Nxc = 4
			Nyc = 4
elif i == 3 or i == 5 or i == 7:
	Nxc = 5
	Nyc = 4
elif i == 4 or i == 6 or i == 8:
	Nxc = 4
	Nyc = 5	

print('=====================================================')
# Load matrix A
subdirectory = subdirectories[0]
filename = "matrix_" + operators[i] + "_" + BC + "_" + eqn
A = np.loadtxt(subdirectory+filename+'.'+data_fileType,unpack=False)
# Plot
figure_title = "Matricial Pattern of the Discrete " + operator_name[i] + " Operator\n on a %ix%i Grid for %s Boundary Conditions" % (Nxc,Nyc,BC)
figure_title_print = "Matricial Pattern of the Discrete " + operator_name[i] + " Operator on a %ix%i Grid for %s Boundary Conditions" % (Nxc,Nyc,BC)
print('Plotting: ' + figure_title_print)
fig = plt.figure(figure_title)
plt.title(figure_title)
elev_min=np.min(A)
elev_max=np.max(A)
mid_val=0
cmap=matplotlib.cm.RdBu_r
plt.imshow(A, cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
if matrix_shape == 'rectangular':
	plt.colorbar(label='Operator Weights',fraction=0.035, pad=0.04)
else:
	plt.colorbar(label='Operator Weights')

if matrix_shape == 'rectangular':
	if eqn == 'ymomentum':
		plt.xticks(np.arange(-0.5,Nxc*Nyc),0*range(0, Nxc*Nyc))
		plt.xlim(-0.5,Nxc*Nyc-0.5)
		plt.yticks(np.arange(-0.5,Nxc*(Nyc-1)),0*range(0, Nxc*(Nyc-1)+1))
	elif eqn == 'ghostCell':
		plt.xticks(np.arange(-0.5,Nxc*(Nyc+1)),0*range(0, Nxc*Nyc))
		plt.xlim(-0.5,Nxc*(Nyc+1)-0.5)
		plt.yticks(np.arange(-0.5,Nxc*Nyc),0*range(0,Nxc*Nyc))
		plt.ylim(Nxc*Nyc-0.5,-0.5)
		# plt.ylim(-0.5,Nxc*(Nyc+1)-0.5)
	else:
		plt.xticks(np.arange(-0.5,Nxc*(Nyc+1)),0*range(0, Nxc*Nyc))
		plt.xlim(-0.5,Nxc*(Nyc+1)-0.5)
		plt.yticks(np.arange(-0.5,Nxc*Nyc),0*range(0, Nxc*Nyc+1))
else:
	plt.xticks(np.arange(-0.5,Nxc*Nyc),0*range(0, Nxc*Nyc+1))
	plt.yticks(np.arange(-0.5,Nxc*Nyc),0*range(0, Nxc*Nyc+1))
plt.grid()
plt.tight_layout()
print('... Saving figure ...')
filename = "SPY_matrix_" + operators[i] + "_" + BC + "_" + eqn
figure_fileType = 'png'
subdirectory = subdirectories[1]
plt.savefig(subdirectory + filename + '.' + figure_fileType,format=figure_fileType,dpi=500)
print('=====================================================')