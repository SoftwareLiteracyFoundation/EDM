#! /usr/bin/env python3

# Python distribution modules
from multiprocessing import Pool
from copy            import deepcopy

# Community modules
from numpy.random import randint, seed
from numpy        import mean, zeros, arange, concatenate
import matplotlib.pyplot as plt

# Local modules
from EDM import EmbedData, FindNeighbors, SimplexProjection, ComputeError, nRow
from ArgParse import ParseCmdLine

#----------------------------------------------------------------------------
# __main__
#----------------------------------------------------------------------------
def CCM():
    '''
    Compute simplex cross-map skill over (random) subsamples of a time series. 

    Data are a .csv file with multiple simultaneous observations (columns)
    and "time" in the first column.  The -r (target) column is used for
    the cross prediction, -c (column) will be embedded to dimension E.

    The -L (libsize) argument specifies a list of library sizes [start,
    stop, increment], -s (subsample) the number of subsamples generated 
    at each library size if -R (replacment) is specified. 

    "Predictions" are made over the same data/embedding slices as the
    library so that -l and -p parameters have no meaning. 
    '''

    args = ParseCmdLine()
    
    if not len( args.columns ) :
        raise RuntimeError( "CCM() -c must specify the column to embed." )
    if not args.target :
        raise RuntimeError( "CCM() -r must specify the target column." )
    
    if args.seed :
        seed( int( args.seed ) )
    
    args.method = 'simplex'
    
    # Save args.plot flag, but disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    # Process pool
    pool = Pool()
    
    # Create iterable with args for the two CrossMap() calls
    argsList = []
    # Cross mapping from -c (columns) to -r (target):
    argsList.append( deepcopy( args ) )

    # Switch the columns and target in args, assume only one column
    target       = args.target
    columns      = args.columns
    args.target  = columns[0]
    args.columns = [target]
                                     
    # Cross mapping from -r (target) to -c (columns) :
    argsList.append( deepcopy( args ) )

    # Switch back for plotting
    args.target  = target
    args.columns = columns
    
    # map_async() : A variant of the map() method that returns a result object
    # of class multiprocessing.pool.AsyncResult
    poolResults = pool.map_async( CrossMap, argsList )

    # Must call AsyncResult.get() to spawn/wait for map_async() results
    results = poolResults.get()

    R0 = results[ 0 ] # tuple ( ID, PredictLibStats{} )
    R1 = results[ 1 ] # tuple ( ID, PredictLibStats{} )

    # Extract results based on the "columns to target" ID
    if R0[ 0 ] in str( args.columns ) + " to " + args.target :
        Col_Targ = R0[ 1 ]
        Targ_Col = R1[ 1 ]
    else :
        Col_Targ = R1[ 1 ]
        Targ_Col = R0[ 1 ]
    
    start, stop, increment = args.libsize

    print( "lib_size  ρ " + args.columns[0] + "   ρ " + args.target )
    print( "            " + args.target     + "     " + args.columns[0] )
    for lib_size in range( start, stop + 1, increment ) :

        col_targ_rho = Col_Targ[ lib_size ][0] # ρ in element [0]
        targ_col_rho = Targ_Col[ lib_size ][0] # ρ in element [0]

        print( "{:>8} {:>10} {:>10}".format( lib_size,
                                           round( col_targ_rho, 2 ),
                                           round( targ_col_rho, 2 ) ) )

    if showPlot :
        #-------------------------------------------------------
        # Plot rho at each subsample
        lib_sizes = arange( start, stop + 1, increment )
        col_targ_rho = [ Col_Targ[ lib_size ][0] \
                         for lib_size in range( start, stop + 1, increment ) ]
        
        targ_col_rho = [ Targ_Col[ lib_size ][0] \
                         for lib_size in range( start, stop + 1, increment ) ]
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        ax.plot( lib_sizes, col_targ_rho,
                 linewidth = 3, color = 'blue',
                 label = args.columns[0] + " to " + args.target )
        ax.plot( lib_sizes, targ_col_rho,
                 linewidth = 3, color = 'red',
                 label = args.target + " to " + args.columns[0] )
        plt.axhline( y = 0, linewidth = 1 )
        ax.legend()
        
        ax.set( xlabel = 'Library Size',
                ylabel = "Cross map correlation " + r' $\rho$',
                title  = args.inputFile +\
                '  E=' + str( args.E  ) )
        plt.show()


#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def CrossMap( args ) :
    
    # Generate embedding on the data to be cross mapped (-c column)
    embedding, colNames, target = EmbedData( args )

    # Use entire library and prediction from embedding matrix
    libraryMatrix = predictionMatrix = embedding
    N_row = nRow( libraryMatrix )

    # Range of library indices
    start, stop, increment = args.libsize
    
    if args.randomLib :
        # Random samples from library with replacement
        maxSamples = args.subsample
    else:
        # Contiguous samples up to the size of the library
        maxSamples = stop
        
    # Simplex: if k_NN not specified, set k_NN to E + 1
    if args.k_NN < 0 :
        args.k_NN = args.E + 1
        if args.verbose:
            print( "CCM() Set k_NN to E + 1 = " + str( args.k_NN ) +\
                   " for SimplexProjection." )

    #----------------------------------------------------------
    # Predictions
    #----------------------------------------------------------
    print( "CCM(): Simplex cross mapping from " + str( args.columns ) +\
           " to " + args.target +  "  E=" + str( args.E ) +\
           " k_nn=" + str( args.k_NN ) +\
           "  Library range: [{}, {}, {}]".format( start, stop, increment ))

    PredictLibStats = {} # { lib_size : ( rho, r, rmse, mae ) }
    # Loop for library sizes
    for lib_size in range( start, stop + 1, increment ) :

        if args.Debug :
            print( "CCM(): lib_size " + str( lib_size ) )

        prediction_rho = zeros( ( maxSamples, 4 ) )
        # Loop for subsamples
        for n in range( maxSamples ) :

            if args.randomLib :
                # Uniform random sample of rows, with replacement
                lib_i = randint( low  = 0,
                                 high = N_row,
                                 size = lib_size )
            else :
                if lib_size >= N_row :
                    # library size exceeded, back down
                    lib_i = arange( 0, N_row )
                    
                    if args.warnings or args.verbose :
                        print( "CCM(): max lib_size is {}, "
                               "lib_size has been limited.".format( N_row ) )
                else :
                    # Contiguous blocks up to N_rows = maxSamples
                    if n + lib_size < N_row :
                        lib_i = arange( n, n + lib_size )
                    else:
                        # n + lib_size exceeds N_row, wrap around to data origin
                        lib_start = arange( n, N_row )
                        max_i     = min( lib_size - (N_row - n), N_row )
                        lib_wrap  = arange( 0, max_i )
                        lib_i     = concatenate( (lib_start, lib_wrap), axis=0)

            #----------------------------------------------------------
            # k_NN nearest neighbors
            #----------------------------------------------------------
            # FindNeighbors will ignore degenerate lib/pred coordinates
            neighbors, distances = FindNeighbors(libraryMatrix   [ lib_i, : ],
                                                 predictionMatrix[ lib_i, : ],
                                                 args )
                
            predictions = SimplexProjection( libraryMatrix[ lib_i, : ],
                                             target       [ lib_i ],
                                             neighbors,
                                             distances,
                                             args )

            rho, r, rmse, mae = ComputeError( target[ lib_i ], predictions )

            prediction_rho[ n, : ] = [ rho, r, rmse, mae ]

        rho_  = mean( prediction_rho[ :, 0 ] )
        r_    = mean( prediction_rho[ :, 1 ] )
        rmse_ = mean( prediction_rho[ :, 2 ] )
        mae_  = mean( prediction_rho[ :, 3 ] )
        
        PredictLibStats[ lib_size ] = ( rho_, r_, rmse_, mae_  )

    # Return tuple with ( ID, PredictLibStats{} )
    return ( str( args.columns ) + " to " + args.target, PredictLibStats )

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    CCM()
