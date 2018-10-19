#! /usr/bin/env python3

# Python distribution modules
from multiprocessing import Pool
from copy            import deepcopy
from collections     import OrderedDict

# Community modules
import matplotlib.pyplot as plt

# Local modules
from ArgParse import ParseCmdLine
from EDM      import EmbedData, ReadEmbeddedData, Prediction

#----------------------------------------------------------------------------
# __main__
#----------------------------------------------------------------------------
def SMapNL():
    '''
    Using ParseCmdLine() arguments, override the -t (theta) to evaluate 
    theta = 0.01 to 9.

    There are two options for data input. One is to use the -c (columns)
    argument so that the -i (inputFile) will be considered a timeseries with 
    embeddings dynamically performed by EmbedData() for each evaluation.  
    The other is to specify -e (embedded) so that -i (inputFile) specifies 
    a .csv file with an embedding or multivariables already in place.  The
    vector in the second column (j=1) will be considered the observed data.
    This will be read by ReadEmbeddedData().
    '''
    
    args = ParseCmdLine()

    args.method = 'smap'

    # Save args.plot flag, but disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    # if -e has not been specified: use EmbedData()
    if not args.embedded :
        embedding, colNames, target = EmbedData( args )
    else :
        # ReadEmbeddedData() sets args.E to the number of columns specified
        # if the -c (columns) and -t (target) options are used, otherwise
        # it uses args.E to read E columns. 
        embedding, colNames, target = ReadEmbeddedData( args )

    # Evaluate theta localization parameter from 0.01 to 9
    Theta = [ 0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9 ]

    # Process pool
    pool = Pool()
    
    # Create iterable with args variants for theta
    argsEmbeddingList = []
    for theta in Theta :
        newArgs       = deepcopy( args )
        newArgs.theta = theta
        # Add the embedding, colNames, target in a tuple
        argsEmbeddingList.append( ( newArgs, embedding, colNames, target ) )

    # map_async() : A variant of the map() method that returns a result object
    # of class multiprocessing.pool.AsyncResult
    poolResults = pool.map_async( PredictFunc, argsEmbeddingList )

    # Must call AsyncResult.get() to spawn/wait for map_async() results
    results = poolResults.get()
    
    # Dict to hold theta : rho pairs from PredictFunc() tuple
    theta_rho = OrderedDict()

    for result in results :
        if result == None:
            continue
        theta_rho[ result[ 0 ] ] = result[ 1 ]

    # Console output
    print( "{:<5} {:<10}".format('θ','ρ') )
    for theta_, rho_ in theta_rho.items():
        print( "{0:<5} {1:<10}".format( theta_, rho_ ) )

    #----------------------------------------------------------
    if showPlot:
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        
        ax.plot( theta_rho.keys(), theta_rho.values(),
                 label = 'Predictions_t(+{0:d})'.format( args.Tp ),
                 color='blue', linewidth = 3 )
        
        ax.set( xlabel = 'S Map Localization θ',
                ylabel = 'Prediction Skill' + r' $\rho$',
                title  = args.inputFile + ' Tp=' + str( args.Tp ) +\
                         ' E=' + str( args.E ) )
        plt.show()

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PredictFunc( argsEmbedding ) :

    args      = argsEmbedding[ 0 ]
    embedding = argsEmbedding[ 1 ]
    colNames  = argsEmbedding[ 2 ]
    target    = argsEmbedding[ 3 ]
    
    rho, rmse, mae, header, output, smap_output = Prediction( embedding,
                                                              colNames,
                                                              target, args )

    return tuple( ( args.theta, round( rho, 4 ) ) )
        
#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    SMapNL()
