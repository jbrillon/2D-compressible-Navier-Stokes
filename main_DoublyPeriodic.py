#=====================================================
# Department of Mechanical Engineering, Purdue University
# ME 614: Computational Fluid Dynamics
# Fall 2018
# Final: 2D Compressible Navier-Stokes
# Julien Brillon
# Python 2.7.15
#=====================================================
# Import libraries
import numpy as np # NumPy: contains basic numerical routines
import scipy # SciPy: contains additional numerical routines to numpy
import matplotlib.pyplot as plt # Matlab-like plotting
import scipy.sparse as scysparse
import scipy.sparse.linalg
#=====================================================
global subdirectories
data_fileType = 'txt'
subdirectories = ['Data/','Figures/']
BCs = ['Periodic']
#*****************************************************
#               GRID SUBROUTINES
#*****************************************************
def gridUniform(lo,hi,N):
    N = np.int(N)
    lo = np.float64(lo)
    hi = np.float64(hi)
    grid = np.linspace(lo,hi,N,dtype=np.float64)
    return grid
#=====================================================
def physicalGrid(lo,hi,Nc):
#+--------------------------------------------------------+
#|    A subroutine that discretizes the spatial domain    |
#+--------------------------------------------------------+
    Np = np.int(Nc+1) # Nc = computational points, Np = physical points
    lo = np.float64(lo)
    hi = np.float64(hi)
    grid = gridUniform(lo,hi,Np) # uniform grid
    return grid 
#=====================================================
def computationalGrid(lo,hi,Nc,prop,component):
    properties = ['pressure', 'x-velocity', 'y-velocity']
    components = ['x','y']
    Nc = np.int(Nc)
    lo = np.float64(lo)
    hi = np.float64(hi)
    XPOINT = physicalGrid(lo,hi,Nc) # physical grid
    Np = np.int(len(XPOINT))
    
    if prop == properties[0]:
        # Ghost cell method
        XPOINT_GHOST = np.empty(Nc+2,dtype=np.float64) # computational grid
        for i in range(0,Np-1):
            XPOINT_GHOST[i+1] = 0.5*(XPOINT[i+1]+XPOINT[i])
            dxStart = XPOINT_GHOST[1] - XPOINT[0]
            dxEnd = XPOINT[Np-1] - XPOINT_GHOST[Nc]
            XPOINT_GHOST[0] = XPOINT[0] - dxStart
            XPOINT_GHOST[Nc+1] = XPOINT[Np-1] + dxEnd

    elif prop == properties[1] and component == components[1]:
        # Ghost cell method
        XPOINT_GHOST = np.empty(Nc+2,dtype=np.float64) # computational grid
        for i in range(0,Np-1):
            XPOINT_GHOST[i+1] = 0.5*(XPOINT[i+1]+XPOINT[i])
            dxStart = XPOINT_GHOST[1] - XPOINT[0]
            dxEnd = XPOINT[Np-1] - XPOINT_GHOST[Nc] 
            XPOINT_GHOST[0] = XPOINT[0] - dxStart
            XPOINT_GHOST[Nc+1] = XPOINT[Np-1] + dxEnd

    elif prop == properties[2] and component == components[0]:
        # Ghost cell method
        XPOINT_GHOST = np.empty(Nc+2,dtype=np.float64) # computational grid
        for i in range(0,Np-1):
            XPOINT_GHOST[i+1] = 0.5*(XPOINT[i+1]+XPOINT[i])
            dxStart = XPOINT_GHOST[1] - XPOINT[0]
            dxEnd = XPOINT[Np-1] - XPOINT_GHOST[Nc] 
            XPOINT_GHOST[0] = XPOINT[0] - dxStart
            XPOINT_GHOST[Nc+1] = XPOINT[Np-1] + dxEnd

    else:
        # Ghost cell method - staggered grid
        ghost_points = 'off'
        if ghost_points == 'on':
            XPOINT_GHOST = np.empty(Np+2,dtype=np.float64) # computational grid
            dxStart = XPOINT[1] - XPOINT[0]
            dxEnd = XPOINT[Np-1] - XPOINT[Np-2]
            XPOINT_GHOST[1:Np+1] = XPOINT # interior points
            XPOINT_GHOST[0] = XPOINT[0] - dxStart # ghost cells
            XPOINT_GHOST[Np+1] = XPOINT[Np-1] + dxEnd # ghost cells
        else: # No ghost points
            XPOINT_GHOST = np.empty(Np,dtype=np.float64) # computational grid
            XPOINT_GHOST = XPOINT # interior points

    return XPOINT_GHOST
#=====================================================
def generateComputationalGrid(xlo,xhi,ylo,yhi,Nxc,Nyc,prop):
    properties = ['pressure', 'x-velocity', 'y-velocity']
    Xc = computationalGrid(xlo,xhi,Nxc,prop,'x') # X with ghost cells
    Yc = computationalGrid(ylo,yhi,Nyc,prop,'y') # Y with ghost cells
    if prop == properties[0]:
        Xng = Xc[1:(len(Xc)-1)] # X with no ghost (ng) cells
        Yng = Yc[1:(len(Yc)-1)] # Y with no ghost (ng) cells
    elif prop == properties[1]:
        Xng = Xc # X with no ghost (ng) cells
        Yng = Yc[1:(len(Yc)-1)] # Y with no ghost (ng) cells
    elif prop == properties[2]:
        Xng = Xc[1:(len(Xc)-1)] # X with no ghost (ng) cells
        Yng = Yc # Y with no ghost (ng) cells

    return (Xc,Yc,Xng,Yng)
#=====================================================
#*****************************************************
#           DISCRETIZATION SUBROUTINES
#*****************************************************
#=====================================================
def ddy(Xc,Yc,BC,eqn):
    if eqn == 'mass' or eqn == 'energy':
        Nxc = int(len(Xc)-2.0)
        Nyc = int(len(Yc)-1.0)
    if eqn == 'xmomentum':
        Nxc = int(len(Xc)-1.0)
        Nyc = int(len(Yc)-1.0)
    if eqn == 'ymomentum':
        Nxc = int(len(Xc)-2.0)
        Nyc = int(len(Yc)-2.0)
    
    A = scysparse.lil_matrix((Nxc*Nyc, Nxc*Nyc), dtype=np.float64)
    for i in range(1,Nxc+1): # x-index / block counter
        if BC == 'Periodic':
            a = scysparse.lil_matrix((Nyc, Nyc), dtype=np.float64)
        for j in range(1,Nyc+1): # y-index
            j_A = Nyc*(i-1)+(j-1) # matrix index
            # Grid spacings
            dyC = Yc[j]-Yc[j-1]
            # Periodic boundary conditions
            if BC == 'Periodic':
                j_a = j-1
                if eqn == 'mass' or eqn == 'energy' or eqn == 'xmomentum':
                    a[j_a,j_a-1] = -1.0/dyC # S
                    a[j_a,j_a] = 1.0/dyC # C
                
                elif eqn == 'ymomentum':
                    a[j_a,j_a] = -1.0/dyC # C
                    if j == Nyc:
                        a[j_a,0] = 1.0/dyC # N
                    else:
                        a[j_a,j_a+1] = 1.0/dyC # N

        if BC == 'Periodic':
            A[(j_A-Nyc+1):j_A+1,(j_A-Nyc+1):j_A+1] = a
    
    A = scysparse.csc_matrix(A)
    return A
#=====================================================
def ddx(Xc,Yc,BC,eqn):

    if eqn == 'mass' or eqn == 'energy':
        Nxc = int(len(Xc)-1.0)
        Nyc = int(len(Yc)-2.0)
    elif eqn == 'xmomentum':
        Nxc = int(len(Xc)-2.0)
        Nyc = int(len(Yc)-2.0)
    if eqn == 'ymomentum':
        Nxc = int(len(Xc)-1.0)
        Nyc = int(len(Yc)-1.0)

    A = scysparse.lil_matrix((Nxc*Nyc, Nxc*Nyc), dtype=np.float64)

    for i in range(1,Nxc+1): # x-index / block counter
        # Grid spacings
        dxC = Xc[i]-Xc[i-1]
        for j in range(1,Nyc+1): # y-index
            j_A = Nyc*(i-1)+(j-1) # matrix index

            # Periodic boundary condtions
            if BC == 'Periodic':
                if eqn == 'mass' or eqn == 'energy' or eqn == 'ymomentum':
                    A[j_A,j_A-Nyc] = -1.0/dxC # W
                    A[j_A,j_A] = 1.0/dxC # C

                elif eqn == 'xmomentum':
                    A[j_A,j_A] = -1.0/dxC # C
                    if i == Nxc:
                        A[j_A,j-1] = 1.0/dxC # E
                    else:
                        A[j_A,j_A+Nyc] = 1.0/dxC # E

    A = scysparse.csc_matrix(A)
    return A
#=====================================================
#*****************************************************
#           INITIALIZATION SUBROUTINES
#*****************************************************
#=====================================================
def P_ratio_init(Xng,Yng):

    sigma = 0.5 # spread
    mu = np.pi # location of max

    fx = (1.0/np.sqrt(2.0*np.pi*sigma**2.0))*np.exp(-((Xng-mu)**2.0)/(2.0*(sigma**2.0)))
    fy = (1.0/np.sqrt(2.0*np.pi*sigma**2.0))*np.exp(-((Yng-mu)**2.0)/(2.0*(sigma**2.0)))
    # fy = np.ones(len(Yng))
    # fx = np.ones(len(Xng))
    # fy = np.zeros(len(Yng)) 
    fxy = np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def plot2D(Xng,Yng,Z,title,filename,cbarlabel):
    global subdirectories

    Nxc = np.int(len(Xng))
    Yxc = np.int(len(Yng))
    X, Y = np.meshgrid(Xng,Yng)
    figure_title = title
    
    figure_title_print = figure_title
    print('Plotting: ' + figure_title_print)
    fig= plt.figure(figure_title)
    plt.title(figure_title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    # Limits
    plt.xlim([xlo,xhi])
    plt.ylim([ylo,yhi])
    if xhi == yhi:
        plt.gca().set_aspect('equal', adjustable='box')
    cs = plt.contourf(X,Y,Z, np.linspace(np.min(Z),np.max(Z),100),cmap='RdPu')
    # cs = plt.contourf(X,Y,Z,cmap='rainbow')
    fig.colorbar(cs,label=cbarlabel)

    # Fix for the white lines between contour levels
    for clr in cs.collections:
        clr.set_edgecolor("face")

    show_grid = 'off'
    if show_grid == 'on':
        Xp = physicalGrid(xlo,xhi,Nxc)
        Yp = physicalGrid(ylo,yhi,Nyc)
        for i in range(0,Nxc+1):
            plt.axvline(Xp[i],linestyle='-',color='k',linewidth=0.25)
        for i in range(0,Nyc+1):
            plt.axhline(Yp[i],linestyle='-',color='k',linewidth=0.25)

    plt.tight_layout()
    print('... Saving figure ...')
    figure_fileType = 'png'
    subdirectory = subdirectories[1]
    plt.savefig(subdirectory + filename + '.' + figure_fileType,format=figure_fileType,dpi=500)
    plt.close()
print('-----------------------------------------------------')
#=====================================================
#*****************************************************
#       TRUNCATION ERROR STUDY SUBROUTINES
#*****************************************************
#=====================================================
def p_xyt_exact(t,Xng,Yng):
    ft = -0.25*np.exp(-4.0*t)
    fx = np.cos(2.0*Xng)
    fy = np.cos(2.0*Yng)
    one = np.ones(len(Xng))
    fxy = ft*(np.outer(one,fx)+np.outer(fy,one)).flatten(order='F')
    return fxy
#=====================================================
def dpdx_exact(t,Xng):
    one = np.ones(len(Xng))
    ft = 0.5*np.exp(-4.0*t)
    fx = np.sin(2.0*Xng)
    fy = one
    fxy = ft*(np.outer(fy,fx)).flatten(order='F')
    return fxy
#=====================================================
def dpdy_exact(t,Yng):
    one = np.ones(len(Yng))
    ft = 0.5*np.exp(-4.0*t)
    fy = np.sin(2.0*Yng)
    fx = one
    fxy = ft*(np.outer(fy,fx)).flatten(order='F')
    return fxy
#=====================================================
def u_xyt_exact(t,Xng,Yng):
    ft = -np.exp(-2.0*t)
    fx = np.cos(Xng)
    fy = np.sin(Yng)
    fxy = ft*np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def v_xyt_exact(t,Xng,Yng):
    ft = np.exp(-2.0*t)
    fx = np.sin(Xng)
    fy = np.cos(Yng)
    fxy = ft*np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def dudx_xyt_exact(t,Xng,Yng):
    ft = np.exp(-2.0*t)
    fx = np.sin(Xng)
    fy = np.sin(Yng)
    fxy = ft*np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def dvdy_xyt_exact(t,Xng,Yng):
    ft = -np.exp(-2.0*t)
    fx = np.sin(Xng)
    fy = np.sin(Yng)
    fxy = ft*np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def rho_u_exact(t,Xng,Yng):
    ft = -np.exp(-2.0*t)
    fx = np.sin(Xng)
    fy = np.sin(Yng)
    fxy = ft*np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def drho_u_dx_exact(t,Xng,Yng):
    ft = -np.exp(-2.0*t)
    fx = np.cos(Xng)
    fy = np.sin(Yng)
    fxy = ft*np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def drho_u_dy_exact(t,Xng,Yng):
    ft = -np.exp(-2.0*t)
    fx = np.sin(Xng)
    fy = np.cos(Yng)
    fxy = ft*np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def dx_xmomentum_exact(t,Xng,Yng):
    ft = np.exp(-4.0*t)
    fx1 = (np.cos(Xng)**2.0)-(np.sin(Xng)**2.0)
    fy1 = np.sin(Yng)**2.0
    fx2 = 0.5*np.sin(2.0*Xng)
    fy2 = np.ones(len(Yng))
    fxy = ft*(np.outer(fy1,fx1) + np.outer(fy2,fx2)).flatten(order='F')
    return fxy
#=====================================================
def dy_xmomentum_exact(t,Xng,Yng):
    ft = -np.exp(-4.0*t)
    fx = np.sin(Xng)**2.0
    fy = np.cos(Yng)**2.0 - np.sin(Yng)**2.0
    fxy = ft*(np.outer(fy,fx)).flatten(order='F')
    return fxy
#=====================================================
def rho_v_exact(t,Xng,Yng):
    ft = -np.exp(-2.0*t)
    fx = np.cos(Xng)
    fy = np.cos(Yng)
    fxy = ft*np.outer(fy,fx).flatten(order='F')
    return fxy
#=====================================================
def dy_ymomentum_exact(t,Xng,Yng):
    ft = 2.0*np.exp(-4.0*t)
    fx1 = np.cos(Xng)*np.sin(Xng)
    fy1 = np.sin(Yng)*np.cos(Yng)
    fy2 = 0.25*np.sin(2.0*Yng)
    fx2 = np.ones(len(Xng))
    fxy = ft*(np.outer(fy1,fx1) + np.outer(fy2,fx2)).flatten(order='F')
    return fxy
#=====================================================
def dx_ymomentum_exact(t,Xng,Yng):
    ft = -2.0*np.exp(-4.0*t)
    fx = np.sin(Xng)*np.cos(Xng)
    fy = np.sin(Yng)*np.cos(Yng)
    fxy = ft*(np.outer(fy,fx)).flatten(order='F')
    return fxy
#=====================================================
def e_xyt_exact(t,Xng,Yng):
    ft = -0.25*np.exp(-4.0*t)
    fx = np.sin(2.0*Xng)
    fy = np.sin(2.0*Yng)
    one = np.ones(len(Xng))
    fxy = ft*(np.outer(one,fx)+np.outer(fy,one)).flatten(order='F')
    return fxy
#=====================================================
def dx_energy_exact(t,Xng,Yng):
    ft = np.exp(-6.0*t)
    term1 = -np.outer(np.sin(Yng),np.sin(Xng)*0.25*np.sin(2.0*Xng))
    term2 = -np.outer(np.sin(Yng),np.sin(Xng)*0.25*np.cos(2.0*Xng))
    term3 = -np.outer(np.sin(Yng)*0.25*np.sin(2.0*Yng),np.sin(Xng))
    term4 = -np.outer(np.sin(Yng)*0.25*np.cos(2.0*Yng),np.sin(Xng))
    term5 = np.outer(np.sin(Yng),0.5*np.cos(Xng)*np.cos(2.0*Xng))
    term6 = -np.outer(np.sin(Yng),0.5*np.cos(Xng)*np.sin(2.0*Xng))
    fxy = ft*(term1+term2+term3+term4+term5+term6).flatten(order='F')
    return fxy
#=====================================================
def dy_energy_exact(t,Xng,Yng):
    ft = np.exp(-6.0*t)
    term1 = np.outer(np.sin(Yng),np.sin(Xng)*0.25*np.sin(2.0*Xng))
    term2 = np.outer(np.sin(Yng),np.sin(Xng)*0.25*np.cos(2.0*Xng))
    term3 = np.outer(np.sin(Yng)*0.25*np.sin(2.0*Yng),np.sin(Xng))
    term4 = np.outer(np.sin(Yng)*0.25*np.cos(2.0*Yng),np.sin(Xng))
    term5 = np.outer(np.cos(Yng)*0.5*np.sin(2.0*Yng),np.sin(Xng))
    term6 = -np.outer(np.cos(Yng)*0.5*np.cos(2.0*Yng),np.sin(Xng))
    fxy = ft*(term1+term2+term3+term4+term5+term6).flatten(order='F')
    return fxy
#=====================================================
def TruncationError(t,Nxc,Nyc,BC,term,component,eqn):
    global xlo, xhi, ylo, yhi

    Nxc = np.int(Nxc)
    Nyc = np.int(Nyc)
    
    # Generate the grids
    properties = ['pressure', 'x-velocity', 'y-velocity']
    Xc_P,Yc_P,Xng_P,Yng_P = generateComputationalGrid(xlo,xhi,ylo,yhi,Nxc,Nyc,properties[0])
    Xc_u,Yc_u,Xng_u,Yng_u = generateComputationalGrid(xlo,xhi,ylo,yhi,Nxc,Nyc,properties[1])
    Xc_v,Yc_v,Xng_v,Yng_v = generateComputationalGrid(xlo,xhi,ylo,yhi,Nxc,Nyc,properties[2])
    
    # Truncation error computation
    if term == 'pressure':
        # Exact field
        P = p_xyt_exact(t,Xng_P,Yng_P)
        if component == 'x':
            exact = dpdx_exact(t,Xng_P)
            approx = ddx(Xc_P,Yc_P,BC).dot(P)
        if component == 'y':
            exact = dpdy_exact(t,Yng_P)
            approx = ddy(Xc_P,Yc_P,BC).dot(P)

    elif term == 'first_derivatives':
        if eqn == 'mass' or eqn == 'energy':
            if component == 'x':
                u = u_xyt_exact(t,Xng_u[1:],Yng_u)
                exact = dudx_xyt_exact(t,Xng_P,Yng_P)
                approx = ddx(Xc_u,Yc_u,BC,eqn).dot(u)
            if component == 'y':
                v = v_xyt_exact(t,Xng_v,Yng_v[1:])
                exact = dvdy_xyt_exact(t,Xng_P,Yng_P)
                approx = ddy(Xc_v,Yc_v,BC,eqn).dot(v)
        elif eqn == 'xmomentum':
            if component == 'x':
                u = u_xyt_exact(t,Xng_P,Yng_u)
                exact = dudx_xyt_exact(t,Xng_u[:(np.int(len(Xng_u))-1)],Yng_u)
                approx = ddx(Xc_P,Yc_u,BC,eqn).dot(u)
            if component == 'y':
                v = v_xyt_exact(t,Xng_u[1:],Yng_v[1:])
                exact = dvdy_xyt_exact(t,Xng_u[1:],Yng_u)
                approx = ddy(Xc_u,Yc_v,BC,eqn).dot(v)
        elif eqn == 'ymomentum':
            if component == 'x':
                u = u_xyt_exact(t,Xng_u[1:],Yng_v[1:])
                exact = dudx_xyt_exact(t,Xng_v,Yng_v[1:])
                approx = ddx(Xc_u,Yc_P,BC,eqn).dot(u)
            if component == 'y':
                v = v_xyt_exact(t,Xng_v,Yng_P)
                exact = dvdy_xyt_exact(t,Xng_v,Yng_v[:(np.int(len(Yng_v))-1)])
                approx = ddy(Xc_v,Yc_P,BC,eqn).dot(v)
    elif term == 'euler':
        rho_u = rho_u_exact(t,Xng_u[1:],Yng_u)
        rho_v = rho_v_exact(t,Xng_v,Yng_v[1:])
        P = p_xyt_exact(t,Xng_P,Yng_P)
        u = u_xyt_exact(t,Xng_u[1:],Yng_u)
        v = v_xyt_exact(t,Xng_v,Yng_v[1:])
        e = e_xyt_exact(t,Xng_P,Yng_P)

        rho_u = np.reshape(rho_u,(Nyc,Nxc),order='F')
        rho_v = np.reshape(rho_v,(Nyc,Nxc),order='F')
        P = np.reshape(P,(Nyc,Nxc),order='F')
        u = np.reshape(u,(Nyc,Nxc),order='F')
        v = np.reshape(v,(Nyc,Nxc),order='F')
        e = np.reshape(e,(Nyc,Nxc),order='F')

        if eqn == 'mass':
            xFLUX = rho_u
            yFLUX = rho_v
            if component == 'x':
                exact = drho_u_dx_exact(t,Xng_P,Yng_P)
                approx = ddx(Xc_u,Yc_u,BC,eqn).dot(xFLUX.flatten(order='F'))
            if component == 'y':
                exact = drho_u_dy_exact(t,Xng_P,Yng_P)
                approx = ddy(Xc_v,Yc_v,BC,eqn).dot(yFLUX.flatten(order='F'))

        elif eqn == 'xmomentum':
            xFLUX = P + (0.5*(rho_u*u + np.roll(rho_u*u,1,axis=1)))
            yFLUX = (0.5*(rho_u + np.roll(rho_u,-1,axis=0)))*(0.5*(v + np.roll(v,-1,axis=1)))
            if component == 'x':
                exact = dx_xmomentum_exact(t,Xng_u[1:],Yng_u)
                approx = ddx(Xc_v,Yc_u,BC,eqn).dot(xFLUX.flatten(order='F'))
            elif component == 'y':
                exact = dy_xmomentum_exact(t,Xng_u[1:],Yng_u)
                approx = ddy(Xc_u,Yc_v,BC,eqn).dot(yFLUX.flatten(order='F'))

        elif eqn == 'ymomentum':
            xFLUX = (0.5*(u + np.roll(u,-1,axis=0)))*(0.5*(rho_v + np.roll(rho_v,-1,axis=1)))
            yFLUX = P + 0.5*(rho_v*v + np.roll(rho_v*v,1,axis=0))
            if component == 'x':
                exact = dx_ymomentum_exact(t,Xng_v,Yng_v[1:])
                approx = ddx(Xc_u,Yc_v,BC,eqn).dot(xFLUX.flatten(order='F'))
            elif component == 'y':
                exact = dy_ymomentum_exact(t,Xng_v,Yng_v[1:])
                approx = ddy(Xc_v,Yc_u,BC,eqn).dot(yFLUX.flatten(order='F'))

        elif eqn == 'energy':
            xFLUX = u*(0.5*(e+P + np.roll(e+P,-1,axis=1)))
            yFLUX = v*(0.5*(e+P + np.roll(e+P,-1,axis=0)))
            if component == 'x':
                exact = dx_energy_exact(t,Xng_P,Yng_P)
                approx = ddx(Xc_u,Yc_u,BC,eqn).dot(xFLUX.flatten(order='F'))
            elif component == 'y':
                exact = dy_energy_exact(t,Xng_P,Yng_P)
                approx = ddy(Xc_v,Yc_v,BC,eqn).dot(yFLUX.flatten(order='F'))

    rmsError = RMSerror(approx-exact)

    if component == 'x':
        dx = Xng_u[1] - Xng_u[0]
        dy = Yng_u[1] - Yng_u[0]
        h = np.sqrt(dx*dy)
    elif component == 'y':
        dx = Xng_v[1] - Xng_v[0]
        dy = Yng_v[1] - Yng_v[0]
        h = np.sqrt(dx*dy)
    return h, rmsError
#=====================================================
def RMSerror(y):
    rms = np.sqrt(np.mean(y**2))
    return rms
#=====================================================
def removeValues(duplicate): 
    final_list = []
    for num in duplicate:
        if num not in final_list:  
            if num>3: # 3 accommodates the discrete operator function
                final_list.append(num)
    return final_list
#=====================================================
#*****************************************************
#              GAS DYNAMIC SUBROUTINES
#*****************************************************
#=====================================================
def viscosity_Sutherland(T):
    # Sutherland parameters
    global T_ref, mu_ref
    S = 110.0 # [K] - Sutherland's temperature
    # Sutherlands viscosity model
    mu = mu_ref*((T/T_ref)**(1.5))*((T_ref + S)/(T + S))
    return mu
#=====================================================
def viscosity_PowerLaw(T):
    # Power law parameters
    global T_ref, mu_ref
    omega = 0.76 # power law parameter
    # Power law viscosity model
    mu = mu_ref*(T/T_ref)**(omega)
    return mu
#=====================================================
#*****************************************************
#               SOLVER SUBROUTINES
#*****************************************************
#=====================================================
def sysEulerEquations(t,Q):
    # Load grids
    global Pgrid,Ugrid,Vgrid
    Xc_P,Yc_P,Xng_P,Yng_P = Pgrid
    Xc_u,Yc_u,Xng_u,Yng_u = Ugrid
    Xc_v,Yc_v,Xng_v,Yng_v = Vgrid
    # Number of points
    Nxc = len(Xng_P)
    Nyc = len(Yng_P)

    # Load spatial operators
    global mass_operators, xmomentum_operators, ymomentum_operators, energy_operators
    ddx_mass, ddy_mass = mass_operators
    ddx_xmomentum, ddy_xmomentum = xmomentum_operators
    ddx_ymomentum, ddy_ymomentum = ymomentum_operators
    ddx_energy, ddy_energy = energy_operators

    # Load conservative quantities
    rho,rho_u,rho_v,e = Q
    
    # Unflatten loaded conservative quantities
    rho = np.reshape(rho,(Nyc,Nxc),order='F')
    rho_u = np.reshape(rho_u,(Nyc,Nxc),order='F')
    rho_v = np.reshape(rho_v,(Nyc,Nxc),order='F')
    e = np.reshape(e,(Nyc,Nxc),order='F')

    # Update flow properties
    global gamma, Pr, mu_ref, R
    u = rho_u/rho
    v = rho_v/rho
    P = (gamma-1.0)*(e - 0.5*rho*(u**2.0 + v**2.0))
    T = P/(rho*R)
    mu = viscosity_Sutherland(T)
    lambda_BVC = -(2.0/3.0)*mu # Bulk viscosity coefficient [Stokes hypothesis]

    #-----------------------------------------------------
    #           (1) Conservation of MASS
    #-----------------------------------------------------
    eqn = 'mass'
    # Governing equation to be integrated
    xFLUX = rho_u
    yFLUX = rho_v
    drho = -ddx_mass.dot(xFLUX.flatten(order='F')) - ddy_mass.dot(yFLUX.flatten(order='F'))
    #-----------------------------------------------------
    #           (2) Conservation of X-MOMENTUM
    #-----------------------------------------------------
    eqn = 'xmomentum'
    # Inviscid adiabatic fluxes
    # x-term
    rho_u2_xFace = (0.5*(rho_u*u + np.roll(rho_u*u,1,axis=1)))
    inviscid_xFLUX = P + rho_u2_xFace
    inviscid_xFLUX = inviscid_xFLUX.flatten(order='F')
    # y-term
    rho_u_yFace = (0.5*(rho_u + np.roll(rho_u,-1,axis=0)))
    v_yFace = (0.5*(v + np.roll(v,-1,axis=1)))
    inviscid_yFLUX = rho_u_yFace*v_yFace
    inviscid_yFLUX = inviscid_yFLUX.flatten(order='F')
    # Viscous fluxes
    # x-term
    viscous_xFLUX = (lambda_BVC + 2.0*mu).flatten(order='F')*ddx_mass.dot(u.flatten(order='F')) + lambda_BVC.flatten(order='F')*ddy_mass.dot(v.flatten(order='F'))
    tau_xx = np.reshape(viscous_xFLUX,(Nyc,Nxc),order='F') # normal stress in x-direction - defined at cell centers
    # y-term
    mu_yFace = (0.5*(mu + np.roll(mu,-1,axis=1)))
    mu_yFace = (0.5*(mu_yFace + np.roll(mu_yFace,-1,axis=0)))
    mu_yFace = mu_yFace.flatten(order='F')
    viscous_yFLUX = mu_yFace*(ddx_xmomentum.dot(v.flatten(order='F')) + ddy_ymomentum.dot(u.flatten(order='F')))
    tau_xy = np.reshape(viscous_yFLUX,(Nyc,Nxc),order='F') # shear stress (xy plane) - defined at cell corners
    # Total fluxes
    xFLUX = -inviscid_xFLUX + viscous_xFLUX
    yFLUX = -inviscid_yFLUX + viscous_yFLUX
    # Rate of change of x-momentum
    drho_u = ddx_xmomentum.dot(xFLUX.flatten(order='F')) + ddy_xmomentum.dot(yFLUX.flatten(order='F'))
    #-----------------------------------------------------
    #           (3) Conservation of Y-MOMENTUM
    #-----------------------------------------------------
    eqn = 'ymomentum'
    # Inviscid adiabatic fluxes
    # x-term
    u_xFace = (0.5*(u + np.roll(u,-1,axis=0)))
    rho_v_xFace = (0.5*(rho_v + np.roll(rho_v,-1,axis=1)))
    inviscid_xFLUX = u_xFace*rho_v_xFace
    inviscid_xFLUX = inviscid_xFLUX.flatten(order='F')
    # y-term
    rho_v2_yFace = 0.5*(rho_v*v + np.roll(rho_v*v,1,axis=0))
    inviscid_yFLUX = P + rho_v2_yFace
    inviscid_yFLUX = inviscid_yFLUX.flatten(order='F')
    # Viscous fluxes
    # x-term
    viscous_xFLUX = tau_xy.flatten(order='F')
    # y-term
    viscous_yFLUX = lambda_BVC.flatten(order='F')*ddx_mass.dot(u.flatten(order='F')) + (lambda_BVC + 2.0*mu).flatten(order='F')*ddy_mass.dot(v.flatten(order='F'))
    tau_yy = np.reshape(viscous_yFLUX,(Nyc,Nxc),order='F') # Normal stress in y-direction - defined at cell centers
    # Total fluxes
    xFLUX = -inviscid_xFLUX + viscous_xFLUX
    yFLUX = -inviscid_yFLUX + viscous_yFLUX
    # Rate of change of y-momentum
    drho_v = ddx_ymomentum.dot(xFLUX) + ddy_ymomentum.dot(yFLUX)
    #-----------------------------------------------------
    #           (4) Conservation of ENERGY
    #-----------------------------------------------------
    eqn = 'energy'
    # Inviscid adiabatic fluxes
    # x - term
    inviscid_xFLUX = u*(0.5*(e+P + np.roll(e+P,-1,axis=1)))
    inviscid_xFLUX = inviscid_xFLUX.flatten(order='F')
    # y - term
    inviscid_yFLUX = v*(0.5*(e+P + np.roll(e+P,-1,axis=0)))
    inviscid_yFLUX = inviscid_yFLUX.flatten(order='F')
    # Conductive fluxes
    # x - term
    conductive_constant = (gamma/(gamma-1.0))*(1.0/Pr)
    mu_xFace = (0.5*(mu + np.roll(mu,-1,axis=1)))
    mu_xFace = mu_xFace.flatten(order='F')
    conductive_xFLUX = conductive_constant*mu_xFace*ddx_xmomentum.dot((P/rho).flatten(order='F'))
    # y - term
    mu_yFace = (0.5*(mu + np.roll(mu,-1,axis=0)))
    mu_yFace = mu_yFace.flatten(order='F')
    conductive_yFLUX = conductive_constant*mu_yFace*ddy_ymomentum.dot((P/rho).flatten(order='F'))
    # Viscous fluxes
    # x - term
    v_xFace = (0.5*(v + np.roll(v,-1,axis=1)))
    v_xFace = (0.5*(v_xFace + np.roll(v_xFace,1,axis=0)))

    tau_xx_xFace = (0.5*(tau_xx + np.roll(tau_xx,-1,axis=1)))
    tau_xy_xFace = (0.5*(tau_xy + np.roll(tau_xy,1,axis=0)))

    viscous_xFLUX = u*tau_xx_xFace + v_xFace*tau_xy_xFace
    viscous_xFLUX = viscous_xFLUX.flatten(order='F')
    # y - term
    u_yFace = (0.5*(u + np.roll(u,1,axis=0)))
    u_yFace = (0.5*(u_yFace + np.roll(u_yFace,-1,axis=1)))
    
    tau_xy_yFace = (0.5*(tau_xy + np.roll(tau_xy,1,axis=1)))
    tau_yy_yFace = (0.5*(tau_yy + np.roll(tau_yy,-1,axis=0)))
    
    viscous_yFLUX = u_yFace*tau_xy_yFace + v*tau_yy_yFace
    viscous_yFLUX = viscous_yFLUX.flatten(order='F')
    # Total fluxes
    xFLUX = -inviscid_xFLUX + conductive_xFLUX + viscous_xFLUX
    yFLUX = -inviscid_yFLUX + conductive_yFLUX + viscous_yFLUX
    # Rate of change of total energy
    de = ddx_energy.dot(xFLUX) + ddy_energy.dot(yFLUX)
    
    #-----------------------------------------------------
    #                       NOTES
    #-----------------------------------------------------
    # Notes on np.roll() use for interpolating FLUXES:
    #   - (-1) denotes forward staggered
    #   - (+1) denotes backward staggered
    #   - axis=1 denotes horizontal
    #   - axis=0 denotes vertical

    return np.array([drho,drho_u,drho_v,de])
#=====================================================
def RK4(h,t,y):
    k1 = h*sysEulerEquations(t, y[:])
    k2 = h*sysEulerEquations(t+0.5*h, y[:]+0.5*k1)
    k3 = h*sysEulerEquations(t+0.5*h, y[:]+0.5*k2)
    k4 = h*sysEulerEquations(t+h, y[:]+k3)
    y[:]= y[:] + (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    return y
#=====================================================
#*****************************************************
#               END OF SUBROUTINES
#*****************************************************
#=====================================================
print('=====================================================')
#=====================================================
#                   GLOBALS
#=====================================================


#=====================================================
#                   SPY PLOT
#-----------------------------------------------------
spy_plot = 'off'
if spy_plot == 'on':

    lo = 0.0
    hi = 2.0*np.pi
    Nxc = 4
    Nyc = 4
    Nxc = np.int(Nxc)
    Nyc = np.int(Nyc)
    
    eqn = 'energy'

    properties = ['pressure', 'x-velocity', 'y-velocity']
    Xc_P,Yc_P,Xng_P,Yng_P = generateComputationalGrid(lo,hi,lo,hi,Nxc,Nyc,properties[0])
    Xc_u,Yc_u,Xng_u,Yng_u = generateComputationalGrid(lo,hi,lo,hi,Nxc,Nyc,properties[1])
    Xc_v,Yc_v,Xng_v,Yng_v = generateComputationalGrid(lo,hi,lo,hi,Nxc,Nyc,properties[2])

    subdirectory = subdirectories[0] + 'DoublyPeriodic/'
    for BC in BCs:
        filename = "matrix_ddx_" + BC + "_" + eqn
        np.savetxt(subdirectory+filename+'.'+data_fileType, ddx(Xc_u,Yc_u,BC,eqn).todense())
        filename = "matrix_ddy_" + BC + "_" + eqn
        np.savetxt(subdirectory+filename+'.'+data_fileType, ddy(Xc_v,Yc_v,BC,eqn).todense())
    
    print('<--- Operator Sparse Matrices Saved --->')
    exit()
#=====================================================
#                   GRID CHECK
#=====================================================
grid_check = 'off'
if grid_check == 'on':
    # Grid parameters
    lo = 0.0
    hi = 10.0#2.0*np.pi
    Nxc = 10
    Nyc = 10
    properties = ['pressure', 'x-velocity', 'y-velocity']
    prop = properties[0]

    Xc,Yc,Xng,Yng = generateComputationalGrid(lo,hi,lo,hi,Nxc,Nyc,prop)
    
    # Plot
    figure_title = 'Computational (red) and Physical (black) grid'
    print('Plotting: ' + figure_title)
    fig = plt.figure(figure_title)
    plt.title(figure_title,fontsize=12,fontweight='bold')
    plt.xlabel(r'$x$',fontsize=12)
    plt.ylabel(r'$y$',fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    Xp = physicalGrid(lo,hi,Nxc)
    Yp = physicalGrid(lo,hi,Nyc)

    for i in range(0,len(Yc)): # with ghost cells
            plt.scatter(Xc,Yc[i]*np.ones(Xc.shape),marker='.',color='r')
    for i in range(0,Nxc+1):
        plt.axvline(Xp[i],linestyle='-',color='k',linewidth=0.5)
    for i in range(0,Nyc+1):
        plt.axhline(Yp[i],linestyle='-',color='k',linewidth=0.5)
    # Contours of physical space
    plt.plot([Xp[0],Xp[0]],[Yp[0],Yp[-1]],linestyle='-',color='k',linewidth=1.5)
    plt.plot([Xp[-1],Xp[-1]],[Yp[0],Yp[-1]],linestyle='-',color='k',linewidth=1.5)
    plt.plot([Xp[0],Xp[-1]],[Yp[0],Yp[0]],linestyle='-',color='k',linewidth=1.5)
    plt.plot([Xp[0],Xp[-1]],[Yp[-1],Yp[-1]],linestyle='-',color='k',linewidth=1.5)
    
    # for i in range(0,Nyc): # without ghost cells
    #   plt.scatter(Xng,Yng[i]*np.ones(Xng.shape),marker='.',color='r')
    plt.tight_layout()
    print('... Saving figure ...')
    figure_name = "grid_" + prop
    figure_fileType = 'png'
    subdirectory = subdirectories[1]
    plt.savefig(subdirectory + 'GRID/' + figure_name + '.' + figure_fileType,format=figure_fileType,dpi=500)
    plt.close()
    exit()
#=====================================================
#           GRID PARAMETERS + GENERATION
#=====================================================
global xlo, xhi, ylo, yhi
BC = BCs[0]
Nxc = 200
Nyc = 200
# Nxc = raw_input('Specify number of x cells: ')
# Nxc = np.int(Nxc)
# Nyc = raw_input('Specify number of y cells: ')
# Nyc = np.int(Nxc)
xlo = 0.0
xhi = 2.0*np.pi
ylo = 0.0
yhi = 2.0*np.pi
# Generate the grids
properties = ['pressure', 'x-velocity', 'y-velocity']
Xc_P,Yc_P,Xng_P,Yng_P = generateComputationalGrid(xlo,xhi,ylo,yhi,Nxc,Nyc,properties[0])
Xc_u,Yc_u,Xng_u,Yng_u = generateComputationalGrid(xlo,xhi,ylo,yhi,Nxc,Nyc,properties[1])
Xc_v,Yc_v,Xng_v,Yng_v = generateComputationalGrid(xlo,xhi,ylo,yhi,Nxc,Nyc,properties[2])
global Pgrid, Ugrid, Vgrid
Pgrid = np.array([Xc_P,Yc_P,Xng_P,Yng_P])
Ugrid = np.array([Xc_u,Yc_u,Xng_u,Yng_u])
Vgrid = np.array([Xc_v,Yc_v,Xng_v,Yng_v])
#=====================================================
#               Truncation Error Study
#-----------------------------------------------------
TE_study = 'off'
if TE_study == 'on':
    #-----------------------------------------------------
    #       Check spatial order of convergence
    #-----------------------------------------------------

    Nx_study = []
    t = 1.0
    term = 'euler'
    component = 'x'
    eqn = 'energy'
    print_rate = 1

    step = 25
    for scale in [1, 1e1, 1e2]:#, 1e3]:#, 1e4]:
        if scale == 1:
            Nx_study.extend([i*scale+1 for i in range(0, 100,step)])
        elif scale == 1e2:
            Nx_study.extend([i*scale+1 for i in range(0, 11,step)])
        else:
            Nx_study.extend([i*scale+1 for i in range(0, 100,step)])
    Nx_study = removeValues(Nx_study)
    Nx_study = np.array(Nx_study)
    if term == 'pressure': # for pressure
        if component == 'x':
            dx_store = (xhi-xlo)/(Nx_study-1.0)
        elif component == 'y':
            dx_store = (yhi-ylo)/(Nx_study-1.0)
    else:
        dx_store = np.zeros(len(Nx_study))

    RMS_store_dx = np.zeros(len(Nx_study))

    # Nt = 50.0 # fixed
    # dt = (t1-t0)/(Nt-1.0)
    # print('Time step dt: %.6f' % dt)
    print('-----------------------------------------------------')
    print('<--- Starting solver runs for all N being studied -->')
    
    for i, Nx in enumerate(Nx_study):
        if np.mod((Nx-1.0),np.float(print_rate))==0.0:
            print('\t Starting solver run for Nx=%i' % Nx)
            
            dx_store[i], RMS_store_dx[i] = TruncationError(t,Nx,Nx,BC,term,component,eqn)
        
        if np.mod((Nx-1.0),np.float(print_rate))==0.0:
            print('\t ... Solver run converged ...')

    print('<--- Solver runs with for all N complete --->')
    print('=====================================================')
    # dx study with dt fixed
    filename = "dx_store_%s_%s_%s" % (term,component,eqn)
    np.savetxt(subdirectories[0]+'DoublyPeriodic/'+filename+'.'+data_fileType, dx_store)
    # RMS of Truncation Error for constant dt
    filename = "RMS_error_dx_%s_%s_%s" % (term,component,eqn)
    np.savetxt(subdirectories[0]+'DoublyPeriodic/'+filename+'.'+data_fileType, RMS_store_dx)
    exit()
    #=====================================================

#=====================================================
#                       MAIN
#=====================================================
#-----------------------------------------------------
#           Constants & Reference Values
#-----------------------------------------------------
global gamma, P_ref, rho_ref, T_ref, Pr, mu_ref, R
# Constants
gamma = 1.4
R = 287.0 # Gas constant
Pr = 0.75 # Prandtl's number
# Reference is Standard ATM at sea level [Anderson Aerodynamics]
P_ref = 101325.0 # [Pa]
T_ref = 288.16 # [K]
rho_ref = P_ref/(R*T_ref) # [kg/m3]
mu_ref = 1.7894e-5 # [kg/(m*s)]
#-----------------------------------------------------
#                   Initialization
#-----------------------------------------------------
print('<--------------- Initializing --------------->')
# Pressure initialization
P0_max = 500.0 # [Pa] - pressure peturbation
P = P_ratio_init(Xng_P,Yng_P) # flattened
P = (P0_max/np.amax(P))*P + P_ref
P = np.reshape(P,(Nyc,Nxc),order='F') # unflattened
# Plot initialization
plot_P_int = 'off'
if plot_P_int == 'on':
    plot2D(Xng_P,Yng_P,P,'Initial Pressure Field','P_init',r'$\delta P$ [Pa]')
    exit()

# Isentropic flow
rho = rho_ref*(P/P_ref)**(1.0/gamma)

# Propagating pulse initialization
c = np.sqrt(gamma*P/rho) # speed of sound
# u = (P0_max/(gamma*P_ref))*c
u = 0.9*c
u_plot_max = 2.0*np.max(c)
u_plot_min = -u_plot_max

# u = np.zeros((Nyc,Nxc))
v = np.zeros((Nyc,Nxc))
rho_u = rho*u
rho_v = rho*v
e = P/(gamma-1.0) + 0.5*rho*(u**2.0 + v**2.0)
Q = np.array([rho.flatten(order='F'),rho_u.flatten(order='F'),rho_v.flatten(order='F'),e.flatten(order='F')])

# Time advancement / numerical stability parameters
ti = 0.0
Nt = 300.0
CFL_c = 0.75
dx = (xhi-xlo)/(Nxc)
dy = (yhi-ylo)/(Nyc)
dt = 0.0 # initialize

# Spatial operators
global mass_operators, xmomentum_operators, ymomentum_operators, energy_operators
mass_operators = np.array([ddx(Xc_u,Yc_u,BC,'mass'),ddy(Xc_v,Yc_v,BC,'mass')])
xmomentum_operators = np.array([ddx(Xc_v,Yc_u,BC,'xmomentum'),ddy(Xc_u,Yc_v,BC,'xmomentum')])
ymomentum_operators = np.array([ddx(Xc_u,Yc_v,BC,'ymomentum'),ddy(Xc_v,Yc_u,BC,'ymomentum')])
energy_operators = np.array([ddx(Xc_u,Yc_u,BC,'energy'),ddy(Xc_v,Yc_v,BC,'energy')])
#-----------------------------------------------------
#               Governing Equations
#-----------------------------------------------------

#-----------------------------------------------------
#        Display solution information
#-----------------------------------------------------
print('- - - - - - - - - - - - - - - - - - - - ')
print('Spatial grid points, Nxc: %i, Nyc: %i' % (np.int(Nxc),np.int(Nyc)))
print('- - - - - - - - - - - - - - - - - - - - ')
print('Spatial step dx: %.6f' % dx)
print('Spatial step dy: %.6f' % dy)
print('Convective-based CFL number: %.3f' % CFL_c)
#-----------------------------------------------------
#                   Simulate
#-----------------------------------------------------
t = ti
n = 0.0
print('... Simulating ...')
while n < Nt:
    
    print('n = %i' % np.int(n))
    print('\tTime step dt: %.6f' % dt)

    # Time advancement
    if n > 0.0:
        Q = RK4(dt,t,Q)

    # Load conservative quantities
    rho,rho_u,rho_v,e = Q
    
    # Unflatten loaded conservative quantities
    rho = np.reshape(rho,(Nyc,Nxc),order='F')
    rho_u = np.reshape(rho_u,(Nyc,Nxc),order='F')
    rho_v = np.reshape(rho_v,(Nyc,Nxc),order='F')
    e = np.reshape(e,(Nyc,Nxc),order='F')

    # Map rho to velocity grids for thermodynamic update
    rho_mappped_to_u = (0.5*(rho + np.roll(rho,-1,axis=1)))
    rho_mappped_to_v = (0.5*(rho + np.roll(rho,-1,axis=0)))

    # Update velocities
    u = rho_u/rho_mappped_to_u
    v = rho_v/rho_mappped_to_v
    
    # Map velocities to e,rho,P grid for thermodynamic update
    u_mapped_to_P = (0.5*(u + np.roll(u,1,axis=1)))
    v_mapped_to_P = (0.5*(v + np.roll(v,1,axis=0)))

    # Thermodynamic update
    P = (gamma-1.0)*(e - 0.5*rho*(u_mapped_to_P**2.0 + v_mapped_to_P**2.0))
    c = np.sqrt(gamma*P/rho) # speed of sound
    vmag = np.sqrt(u**2.0 + v**2.0)

    # Time step update
    dt = CFL_c*(np.min([dx,dy])/(np.max(vmag)+np.max(c)))

    # Step is complete, now update time step counter for next step
    t = t + dt
    n = n + 1.0

    plot_contour = 'on'
    if plot_contour == 'on':
        cbarlabel = r"$\delta P$ [Pa]"
        figure_title = "Pressure Field, Time step = %i" % np.int(n)
        print('Plotting: ' + figure_title)
        fig = plt.figure(figure_title)
        plt.title(figure_title,fontsize=12)#,fontweight='bold')
        plt.xlabel(r'x',fontsize=12)
        plt.ylabel(r"y",rotation=90,fontsize=12)
        plt.xlim([xlo,xhi])
        plt.ylim([ylo,yhi])
        plt.tight_layout()

        Z = P
        X = Xng_P
        Y = Yng_P
        if xhi == yhi:
            plt.gca().set_aspect('equal', adjustable='box')
        # cs = plt.contourf(X,Y,Z, np.linspace(np.min(Z),np.max(Z),50),cmap='RdPu')
        cs = plt.contourf(X,Y,Z-P_ref, np.linspace(-P0_max,P0_max,50),cmap='seismic')
        # cs = plt.contourf(X,Y,Z,cmap='rainbow')
        fig.colorbar(cs,label=cbarlabel)

        # Fix for the white lines between contour levels
        for clr in cs.collections:
            clr.set_edgecolor("face")

        print('... Saving figure ...')
        if np.int(n) < 10:
            filename = 'P_field_tstep_000%i' % np.int(n)    
        elif np.int(n) < 100:
            filename = 'P_field_tstep_00%i' % np.int(n)
        elif np.int(n) < 1000:
            filename = 'P_field_tstep_0%i' % np.int(n)
        else:
            filename = 'P_field_tstep_%i' % np.int(n)
        figure_fileType = 'png'
        subdirectory = subdirectories[1]+'DoublyPeriodic/'
        plt.savefig(subdirectory + 'PressureContour/' + filename + '.' + figure_fileType,format=figure_fileType,dpi=100)
        plt.close()
    
    plot_slice = 'on'
    if plot_slice == 'on':
        slice_index = 105
        figure_title = "Pressure vs x at y=%.3f, Time step = %i" % (Yng_P[slice_index],np.int(n))
        print('Plotting: ' + figure_title)
        fig = plt.figure(figure_title)
        plt.title(figure_title,fontsize=12)#,fontweight='bold')
        plt.xlabel(r'x',fontsize=12)
        plt.ylabel(r"$\delta P$ [Pa]",rotation=90,fontsize=12)
        plt.xlim([xlo,xhi])
        plt.ylim([-P0_max,P0_max])
        plt.tight_layout()

        plt.plot(Xng_P,P[slice_index,:]-P_ref)

        print('... Saving figure ...')
        if np.int(n) < 10:
            filename = 'P_vs_x_tstep_000%i' % np.int(n) 
        elif np.int(n) < 100:
            filename = 'P_vs_x_tstep_00%i' % np.int(n)
        elif np.int(n) < 1000:
            filename = 'P_vs_x_tstep_0%i' % np.int(n)
        else:
            filename = 'P_vs_x_tstep_%i' % np.int(n)
        figure_fileType = 'png'
        subdirectory = subdirectories[1]+'DoublyPeriodic/'
        plt.savefig(subdirectory + 'PressureSlice/' + filename + '.' + figure_fileType,format=figure_fileType,dpi=100)
        plt.close()

    plot_u_velocity_slice = 'on'
    if plot_u_velocity_slice == 'on':
        slice_index = 105
        figure_title = "u vs x at y=%.3f, Time step = %i" % (Yng_u[slice_index],np.int(n))
        print('Plotting: ' + figure_title)
        fig = plt.figure(figure_title)
        plt.title(figure_title,fontsize=12)#,fontweight='bold')
        plt.xlabel(r'x',fontsize=12)
        plt.ylabel(r"u [m/s]",rotation=90,fontsize=12)
        plt.xlim([xlo,xhi])
        plt.ylim([u_plot_min,u_plot_max])
        plt.tight_layout()

        plt.plot(Xng_u[1:],u[slice_index,:])

        print('... Saving figure ...')
        if np.int(n) < 10:
            filename = 'u_vs_x_tstep_000%i' % np.int(n) 
        elif np.int(n) < 100:
            filename = 'u_vs_x_tstep_00%i' % np.int(n)
        elif np.int(n) < 1000:
            filename = 'u_vs_x_tstep_0%i' % np.int(n)   
        else:
            filename = 'u_vs_x_tstep_%i' % np.int(n)
        figure_fileType = 'png'
        subdirectory = subdirectories[1]+'DoublyPeriodic/'
        plt.savefig(subdirectory + 'uVelocity/' + filename + '.' + figure_fileType,format=figure_fileType,dpi=100)
        plt.close()

    plot_energy_slice = 'off'
    if plot_energy_slice == 'on':
        slice_index = 105
        figure_title = "Energy vs x at y=%.3f, Time step = %i" % (Yng_P[slice_index],np.int(n))
        print('Plotting: ' + figure_title)
        fig = plt.figure(figure_title)
        plt.title(figure_title,fontsize=12)#,fontweight='bold')
        plt.xlabel(r'x',fontsize=12)
        plt.ylabel(r"Total Energy [J]",rotation=90,fontsize=12)
        plt.xlim([xlo,xhi])
        plt.ylim([-100,100])
        plt.tight_layout()

        plt.plot(Xng_P,e[slice_index,:])

        print('... Saving figure ...')
        if np.int(n) < 10:
            filename = 'energy_vs_x_tstep_000%i' % np.int(n)    
        elif np.int(n) < 100:
            filename = 'energy_vs_x_tstep_00%i' % np.int(n)
        elif np.int(n) < 1000:
            filename = 'energy_vs_x_tstep_0%i' % np.int(n)
        else:
            filename = 'energy_vs_x_tstep_%i' % np.int(n)
        figure_fileType = 'png'
        subdirectory = subdirectories[1]+'DoublyPeriodic/'
        plt.savefig(subdirectory + 'Energy/' + filename + '.' + figure_fileType,format=figure_fileType,dpi=100)
        plt.close()

print('<--- Simulation Finished. --->')
print('-----------------------------------------------------')
print('=====================================================')