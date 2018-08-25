
from argparse import ArgumentParser

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = ArgumentParser( description = 'EDM' )
    
    parser.add_argument('-m', '--method',
                        dest   = 'method', type = str, 
                        action = 'store', default = 'Simplex',
                        help = 'Type of projection Simplex or SMap.')

    parser.add_argument('-p', '--prediction', nargs = '+',
                        dest   = 'prediction', type = int, 
                        action = 'store', default = [1, 10],
                        help = 'Prediction start/stop indices.')

    parser.add_argument('-l', '--library', nargs = '+',
                        dest   = 'library', type = int, 
                        action = 'store', default = [1, 10],
                        help = 'Library start/stop indices.')

    parser.add_argument('-E', '--EmbedDimension',
                        dest   = 'E', type = int, 
                        action = 'store', default = -1,
                        help = 'Embedding dimension.')

    parser.add_argument('-k', '--knn',
                        dest   = 'k_NN', type = int, 
                        action = 'store', default = -1,
                        help = 'Number of nearest neighbors.')

    parser.add_argument('-T', '--Tp',
                        dest   = 'Tp', type = int, 
                        action = 'store', default = 0,
                        help = 'Forecast interval (0 default).')

    parser.add_argument('-t', '--theta',
                        dest   = 'theta', type = float, 
                        action = 'store', default = 0,
                        help = 'S-Map local weighting exponent (0 default).')

    parser.add_argument('-M', '--multiview',
                        dest   = 'multiview', type = int, 
                        action = 'store', default = 0,
                        help = 'Multiview ensemble size (sqrt(m) default).')

    parser.add_argument('-u', '--tau',
                        dest   = 'tau', type = int, 
                        action = 'store', default = 1,
                        help = 'Time delay (tau).')

    parser.add_argument('-f', '--forwardTau',
                        dest   = 'forwardTau', 
                        action = 'store_true', default = False,
                        help = 'Embed as t + tau instead of t - tau.')

    parser.add_argument('-c', '--columns', nargs = '*',
                        dest   = 'columns', type = str,
                        action = 'store', default = '',
                        help = 'Data or embedded data column names.')

    parser.add_argument('-r', '--target',
                        dest   = 'target', type = str,
                        action = 'store', default = None,
                        help = 'Data library target column name.')

    parser.add_argument('-e', '--embedded',
                        dest   = 'embedded',
                        action = 'store_true', default = False,
                        help = 'Input data is an embedding.')

    parser.add_argument('-L', '--libsize', nargs = '*',
                        dest   = 'libsize', type = int,
                        action = 'store',
                        default = [ 10, 80, 10 ],
                        help = 'CCM Library size range [start, stop, incr].')

    parser.add_argument('-s', '--subsample',
                        dest   = 'subsample', type = int, 
                        action = 'store',      default = 100,
                        help = 'Number subsamples generated at each library.')

    parser.add_argument('-R', '--randomLib',
                        dest   = 'randomLib', 
                        action = 'store_true', default = False,
                        help = 'CCM random library samples enabled.')

    parser.add_argument('-S', '--seed',
                        dest   = 'seed', type = int, 
                        action = 'store',      default = None,
                        help = 'Random number generator seed: (None default)')

    parser.add_argument('-pa', '--path',
                        dest   = 'path', type = str, 
                        action = 'store',      default = './data/',
                        help = 'Input & Output file path.')

    parser.add_argument('-i', '--inputFile', required = True,
                        dest   = 'inputFile', type = str, 
                        action = 'store',     default = None,
                        help = 'Input observation file.')

    parser.add_argument('-o', '--outputFile',
                        dest   = 'outputFile', type = str, 
                        action = 'store',      default = None,
                        help = 'Output prediction file.')
    
    parser.add_argument('-oe', '--outputEmbed',
                        dest   = 'outputEmbed', type = str, 
                        action = 'store',      default = None,
                        help = 'Output embedded data file.')
    
    parser.add_argument('-fs', '--figureSize', nargs = 2,
                        dest   = 'figureSize', type = float,
                        action = 'store', default = [ 6.4, 4 ],
                        help = 'Figure size (default [6.4, 4]).')
    
    parser.add_argument('-P', '--plot',
                        dest   = 'plot',
                        action = 'store_true', default = False,
                        help = 'Show plot.')
    
    parser.add_argument('-PX', '--plotXLabel',
                        dest   = 'plotXLabel', type = str,
                        action = 'store', default = 'Time ()',
                        help = 'Plot x-axis label.')
    
    parser.add_argument('-PY', '--plotYLabel',
                        dest   = 'plotYLabel', type = str,
                        action = 'store', default = 'Amplitude ()',
                        help = 'Plot y-axis label.')
    
    parser.add_argument('-PD', '--plotDate',  # Set automatically 
                        dest   = 'plotDate',
                        action = 'store_true', default = False,
                        help = 'Time values are pyplot datetime numbers.')
    
    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose',
                        action = 'store_true', default = False,
                        help = 'Print status messages.')
    
    parser.add_argument('-w', '--warnings',
                        dest   = 'warnings',
                        action = 'store_true', default = False,
                        help = 'Show warnings.')
    
    parser.add_argument('-D', '--Debug',
                        dest   = 'Debug',
                        action = 'store_true', default = False,
                        help = 'Activate Debug messsages.')
    
    args = parser.parse_args()

    # If S-Map prediction, require k_NN > E + 1, default is all neighbors.
    # If Simplex and k_NN not specified, k_NN set to E+1 in Prediction()
    if "smap" in args.method.lower() and args.k_NN > 0:
        if args.k_NN <= args.E :
            raise RuntimeError( "ParseCmdLine() k_NN must be at least E+1 " +\
                                " with method SMap." )
        
    # Convert library and prediction indices to zero-offset
    args.prediction = [ x-1 for x in args.prediction ]
    args.library    = [ x-1 for x in args.library    ]
    
    # Python slice indexing start:stop:increment means < stop
    # so [0:9:1] returns 9 elements from 0 to 8, not 10 from 0 to 9
    args.prediction[-1] = args.prediction[-1] + 1
    args.library   [-1] = args.library   [-1] + 1

    # Convert figureSize to a tuple
    args.figureSize = tuple( args.figureSize )

    if args.Debug:
        print( 'ParseCmdLine()' )
        print( args )

    return args
