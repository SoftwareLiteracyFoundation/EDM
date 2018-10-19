#! /usr/bin/env python3

from multiprocessing import Pool
from copy            import deepcopy

import matplotlib.pyplot as plt

from ArgParse import ParseCmdLine
from EDM      import EmbedData, ReadEmbeddedData, Prediction

#----------------------------------------------------------------------------
# __main__
#----------------------------------------------------------------------------
def PredictDecay():
    '''
    Using ParseCmdLine() arguments, override Tp to evaluate Tp = 1 to 10.

    There are two options for data input. One is to use the -c (columns)
    argument so that the -i (inputFile) will be considered a timeseries with 
    embeddings dynamically performed by EmbedData() for each evaluation.  
    The other is to specify -e (embedded) so that -i (inputFile) specifies 
    a .csv file with an embedding or multivariables already in place.  The
    vector in the second column (j=1) will be considered the observed data.
    This will be read by ReadEmbeddedData().

    Prediction() sets k_NN equal to E+1 if -k not specified and method 
    is Simplex.
    '''
    
    args = ParseCmdLine()

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

    # Process pool
    pool = Pool()
    
    # Create iterable with args variants for Tp = 1 to 10
    argsEmbeddingList = []
    for T in range( 1, 11 ) :
        newArgs    = deepcopy( args )
        newArgs.Tp = T
        # Add the embedding, colNames, target in a tuple
        argsEmbeddingList.append( ( newArgs, embedding, colNames, target ) )

    # map_async() : A variant of the map() method that returns a result object
    # of class multiprocessing.pool.AsyncResult
    poolResults = pool.map_async( PredictFunc, argsEmbeddingList )

    # Must call AsyncResult.get() to spawn/wait for map_async() results
    results = poolResults.get()
    
    Tp_rho = {} # Dict to hold Tp : rho pairs from PredictFunc() tuple

    for result in results :
        if result == None:
            continue
        
        Tp_rho[ result[ 0 ] ] = result[ 1 ]

    # Console output
    print( "{:<5} {:<10}".format('Tp','ρ') )
    for T_, rho_ in Tp_rho.items():
        print( "{0:<5} {1:<10}".format( T_, rho_ ) )

    #----------------------------------------------------------
    if showPlot:
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        
        ax.plot( Tp_rho.keys(), Tp_rho.values(),
                 label = 'Predictions_t(+{0:d})'.format( args.Tp ),
                 color='blue', linewidth = 3 )
        
        ax.set( xlabel = 'Forecast time Tp',
                ylabel = 'Prediction Skill' + r' $\rho$',
                title  = args.inputFile +\
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
    return tuple( ( args.Tp, round( rho, 3 ) ) )
        
#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    PredictDecay()
