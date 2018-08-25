#! /usr/bin/env python3

from argparse import ArgumentParser

from   scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------------------------------------
# Main module
#----------------------------------------------------------------------------
def main():
    '''
    Create, plot and write out multidimensional data from the Lorenz '96
    dynamical model.

    From Lorenz '96:

    K variables X_1, ... X_k, governed by K equations:
           d X_k / dt = -X_k-2 * X_k-1 + X_k-1 * X_k+1 - X_k + F

    We assume K > 3; the equations are of little interest otherwise.

    For very small values of F, all solutions converge to the steady state
    solution X = F. For larger F most solutions are periodic, but for still
    larger values of F (dependent on K) chaos ensues. For K = 36 and F = 8
    λ_1 corresponds to a doubling time of 2.1 days. If F is 10 the time drops
    to 1.5 days. ... this scaling makes time unit equal to 5 days. With a
    time step of Δt = 0.05 units, or 6 hours. 

    --
    Lorenz, Edward (1996). Predictability – A problem partly solved,
    Seminar on Predictability, Vol. I, ECMWF
    '''
    
    args = ParseCmdLine()
    
    Lorenz96( args )
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def Lorenz96( args ):
    # initial state (equilibrium)
    x0 = args.forceConstant * np.ones( args.nVariables )

    x0[ args.i_perturb ] += args.perturb # add small perturbation

    t = np.arange( 0.0, args.T, args.dT )

    # odeint requires "extra" variables to be in a tuple with name matching
    # inside the derivative function, so make N, F explicit
    N = args.nVariables
    F = args.forceConstant
    x = odeint( dLorenz96, x0, t, args = (N, F) )

    # Create output matrix
    # get index of starting point to exclude transient start
    exclude = int( np.where( t == args.exclude )[0] )
    
    # t is an array, first cast as matrix and transpose to merge with x
    output = np.concatenate( ( np.asmatrix( t ).T, x ), 1 )
    output = output[ exclude::, ]

    if args.outputFile:
      header = 'Time'
      for dim in range( args.nVariables ) :
        header = header + ',V' + str( dim + 1 )

      np.savetxt( args.outputFile, output, fmt = '%.4f', delimiter = ',',
                  header = header, comments = '' )

    #------------------------------------------------------------
    # 3-D plot first three variables in args.dimensions
    if '3D' in args.plot :
      from mpl_toolkits.mplot3d import Axes3D
      fig3D = plt.figure()
      ax3D  = fig3D.gca(projection='3d')
      
      ax3D.plot( x[ exclude::, args.dimensions[0] ],
                 x[ exclude::, args.dimensions[1] ],
                 x[ exclude::, args.dimensions[2] ] )
      
      ax3D.set_xlabel( '$x_{0:d}$'.format( args.dimensions[0] ) )
      ax3D.set_ylabel( '$x_{0:d}$'.format( args.dimensions[1] ) )
      ax3D.set_zlabel( '$x_{0:d}$'.format( args.dimensions[2] ) )
      plt.show()
    
    #------------------------------------------------------------
    # 2-D plot all variables in args.dimensions
    elif '2D' in args.plot :
      plotColors = [ 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 
                     'blue', 'green', 'red', 'cyan', 'magenta', 'black' ]
      fig, ax = plt.subplots()
      
      for d in args.dimensions:
        ax.plot( output[:,0], output[:,d],
                 color = plotColors[d], linewidth = 3 )

      ax.set( xlabel = 'index ()',
              ylabel = 'amplitude ()',
              title  = 'Lorenz 96' )
      plt.show()

#----------------------------------------------------------------------------
#  State derivatives
#  Relies on Python array wrapping for negative/overflow indicies
#----------------------------------------------------------------------------
def dLorenz96( x, t, N, F ):

  d = np.zeros(N)
  
  # first the 3 edge cases: i=1,2,N
  d[0]   = (x[1] - x[N-2]) * x[N-1] - x[0]
  d[1]   = (x[2] - x[N-1]) * x[0]   - x[1]
  d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
  
  # then the general case
  for i in range(2, N-1):
      d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
      
  # add the forcing term
  d = d + F

  return d

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = ArgumentParser( description = 'Lorenz 96' )
    
    parser.add_argument('-D', '--nVariables',
                        dest   = 'nVariables', type = int, 
                        action = 'store',     default = 5,
                        help = 'Number of variables.')
    
    parser.add_argument('-f', '--forceConstant',
                        dest   = 'forceConstant', type = int, 
                        action = 'store',      default = 8,
                        help = 'Forcing constant.')

    parser.add_argument('-p', '--perturb',
                        dest   = 'perturb', type = float, 
                        action = 'store',      default = 0.01,
                        help = 'Pertubation value.')

    parser.add_argument('-i', '--iPerturb',
                        dest   = 'i_perturb', type = int, 
                        action = 'store',      default = 3,
                        help = 'Pertubation index.')

    parser.add_argument('-T', '--T',
                        dest   = 'T', type = float, 
                        action = 'store',      default = 60.,
                        help = 'Max time.')

    parser.add_argument('-t', '--dT',
                        dest   = 'dT', type = float, 
                        action = 'store',      default = 0.05,
                        help = 'Time increment.')

    parser.add_argument('-x', '--exclude',
                        dest   = 'exclude', type = float, 
                        action = 'store',      default = 10.,
                        help = 'Initial transient time to exclude.')

    parser.add_argument('-o', '--outputFile',
                        dest   = 'outputFile', type = str, 
                        action = 'store', default = None,
                        help = 'Output file.')

    parser.add_argument('-P', '--plot',
                        dest   = 'plot', type = str,
                        action = 'store',      default = '3D',
                        help = '2D or 3D plot')

    parser.add_argument('-d', '--dimensions', nargs='+',
                        dest   = 'dimensions', type = int, 
                        action = 'store', default = [1, 2, 3],
                        help = 'Dimensions to 2D plot.')

    args = parser.parse_args()

    if args.i_perturb >= args.nVariables :
        print( "i_perturb > D, setting to 1" )
        args.i_perturb = 1

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation
# main() will only be called 'automatically' when the script is passed
# as the argument to the Python interpreter, not when imported as a module.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
