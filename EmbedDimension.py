#! /usr/bin/env python3

from multiprocessing import Pool
from copy            import deepcopy

import matplotlib.pyplot as plt

from ArgParse import ParseCmdLine
from EDM      import EmbedData, ReadEmbeddedData, Prediction

#----------------------------------------------------------------------------
# __main__
#----------------------------------------------------------------------------
def EmbedDimension():
    '''
    Using ParseCmdLine() arguments, override E and k_NN to evaluate 
    embeddings for E = 1 to 10.

    There are two options for data input. One is to use the -c (columns)
    argument so that the -i (inputFile) will be considered a timeseries with 
    embeddings dynamically performed by EmbedData() for each evaluation.  
    The other is to specify -e (embedded) so that -i (inputFile) specifies 
    a .csv file with an embedding or multivariables already in place.  The
    vector in the second column (j=1) will be considered the observed data.
    This will be read by ReadEmbeddedData().

    Prediction() sets k_NN equal to E + 1 if -k not specified and method 
    is Simplex.
    '''
    
    args = ParseCmdLine()

    # Save args.plot flag, but disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    # Process pool
    pool = Pool()
    
    # Create iterable with args variants for E = 1 to 10
    argsList = []
    for E in range( 1, 11 ) :
        newArgs      = deepcopy( args )
        newArgs.E    = E
        newArgs.k_NN = E + 1
        argsList.append( newArgs )

    # Submit EmbedPredict jobs to the process pool
    results = pool.map( EmbedPredict, argsList )
    
    E_rho = {} # Dict to hold E : rho pairs from EmbedPredict() tuple

    for result in results :
        if result == None:
            continue
        
        E_rho[ result[ 0 ] ] = result[ 1 ]

    # Console output
    print( "{:<5} {:<10}".format('E','ρ') )
    for E_, rho_ in E_rho.items():
        print( "{0:<5} {1:<10}".format( E_, rho_ ) )

    #----------------------------------------------------------
    if showPlot:
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        
        ax.plot( E_rho.keys(), E_rho.values(),
                 label = 'Predictions_t(+{0:d})'.format( args.Tp ),
                 color='blue', linewidth = 3 )
        
        ax.set( xlabel = 'Embedding Dimension',
                ylabel = 'Prediction Skill' + r' $\rho$',
                title  = args.inputFile +\
                         ' Tp=' + str( args.Tp ) )
        plt.show()
    
#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def EmbedPredict( args ):

    # if -e has been specified: use ReadEmbeddedData()
    # ReadEmbeddedData() sets args.E to the number of columns specified
    # if the -c (columns) and -t (target) options are used, otherwise
    # it uses args.E to read E columns.
    if args.embedded :
        # Set args.E so at least 10 dimensions are read.
        E      = args.E
        args.E = 10
        embedding, colNames, target = ReadEmbeddedData( args )
        # Reset args.E for Prediction
        args.E = E
    
    else :
        # -e not specified, embed on each iteration
        embedding, colNames, target = EmbedData( args )
        
    rho, rmse, mae, header, output, smap_output = Prediction( embedding,
                                                              colNames,
                                                              target, args )
    return tuple( ( args.E, round( rho, 3 ) ) )

    
#----------------------------------------------------------------------------
# Provide for cmd line invocation anc clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    EmbedDimension()
