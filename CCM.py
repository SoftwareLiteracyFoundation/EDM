#! /usr/bin/env python3

# Python distribution modules
from multiprocessing import Pool
from copy            import deepcopy

# Community modules
from numpy.random  import randint, seed
from numpy         import mean, zeros, arange, concatenate, \
                   round, amax, full, unique
import matplotlib.pyplot as plt

# Local modules
from EDM import EmbedData, SimplexProjection, \
                ComputeError, nRow, Distance, DistanceMetric
from ArgParse import ParseCmdLine

#----------------------------------------------------------------------------
# __main__
#----------------------------------------------------------------------------
def CCM():
    '''
    Compute simplex cross-map skill over (random) subsamples of a time series. 

    Data are a .csv file with multiple simultaneous observations (columns)
    and "time" in the first column.  The -r (target) column is used for
    the cross prediction, -c (column) is embedded to dimension E.
    CCM is performed simultaneously between both target and column with
    the use of a process Pool. 
    
    Arguments: 
    -L (libsize) specifies a list of library sizes [start, stop, increment]
    -s (subsample) number of subsamples generated at each library size, if:
    -R (replacement) subsample with replacement. 

    Simplex "Predictions" are made over the same data/embedding slices as
    the library so that -l and -p parameters have no meaning. 
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
    
    # Submit the CrossMap jobs to the process pool
    results = pool.map( CrossMap, argsList )

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

    # Range of CCM library indices
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

    #-----------------------------------------------------------------
    print( "CCM(): Simplex cross mapping from " + str( args.columns ) +\
           " to " + args.target +  "  E=" + str( args.E ) +\
           " k_nn=" + str( args.k_NN ) +\
           "  Library range: [{}, {}, {}]".format( start, stop, increment ))

    #-----------------------------------------------------------------
    # Distance for all possible pred : lib E-dimensional vector pairs
    # Distances is a Matrix of all row to to row distances
    #-----------------------------------------------------------------
    Distances = GetDistances( libraryMatrix, args )
    
    #----------------------------------------------------------
    # Predictions
    #----------------------------------------------------------
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
            # k_NN nearest neighbors : Local GetNeighbors() function
            #----------------------------------------------------------
            neighbors, distances = GetNeighbors( Distances, lib_i, args )

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
# 
#----------------------------------------------------------------------------
def GetDistances( libraryMatrix, args ) :
    '''
    Note that for CCM the libraryMatrix and predictionMatrix are the same.

    Return Distances: a square matrix with distances.
    Matrix elements D[i,j] hold the distance between the E-dimensional
    phase space point (vector) between rows (observations) i and j.
    '''
    
    N_row = nRow( libraryMatrix )
    
    D = full( (N_row, N_row), 1E30 ) # Distance matrix init to 1E30
    E = args.E + 1
    
    for row in range( N_row ) :
        # Get E-dimensional vector from this library row
        # Exclude the 1st column (j=0) of times
        y = libraryMatrix[ row, 1:E: ]

        for col in range( N_row ) :
            # Ignore the diagonal (row == col)
            if row == col :
                continue
            
            # Find distance between vector (y) and other library vector
            # Exclude the 1st column (j=0) of Time
            D[ row, col ] = Distance( libraryMatrix[ col, 1:E: ], y )
            # Insert degenerate values since D[i,j] = D[j,i]
            D[ col, row ] = D[ row, col ]

    return ( D )
    
#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def GetNeighbors( Distances, lib_i, args ) :
    '''
    Return a tuple of ( neighbors, distances ). neighbors is a matrix of 
    row indices in the library matrix. Each neighbors row represents one 
    prediction vector. Columns are the indices of k_NN nearest neighbors 
    for the prediction vector (phase-space point) in the library matrix.
    distances is a matrix with the same shape as neighbors holding the 
    corresponding distance values in each row.

    Note that the indices in neighbors are not the original indices in
    the libraryMatrix rows (observations), but are with respect to the
    distances subset defined by the list of rows lib_i, and so have values
    from 0 to len(lib_i)-1.
    '''
    
    N_row = len( lib_i )           # Subset of libraryMatrix and Distances
    col_i = arange( len( lib_i ) ) # Vector of col indices [0,...len(lib_i)-1]
    k_NN  = args.k_NN

    if args.Debug :
        print( 'GetNeighbors() Distances:' )
        print( round( Distances[ 0:5, 0:5 ], 4 ) )
        print( 'N_row = ' + str( N_row ) )

    # Matrix to hold libraryMatrix row indices
    # One row for each prediction vector, k_NN columns for each index
    neighbors = zeros( (N_row, k_NN), dtype = int )

    # Matrix to hold libraryMatrix k_NN distance values
    # One row for each prediction vector, k_NN columns for each index
    distances = zeros( (N_row, k_NN) )

    # For each prediction vector (row in predictionMatrix) find the list
    # of library indices that are within k_NN points
    row = 0
    for row_i in lib_i :
        # Take D[ row, col ] a row at a time, col represent other row distance
        # Sort based on Distance with paired column indices
        # D_row_i is a list of tuples sorted by increasing distance
        D_row_i = sorted( zip( Distances[ row_i, lib_i ], col_i ) )

        # Take the first k_NN distances and column indices
        k_NN_distances = [ x[0] for x in D_row_i ][ 0:k_NN ] # distance
        k_NN_neighbors = [ x[1] for x in D_row_i ][ 0:k_NN ] # index

        if args.Debug :
            if amax( k_NN_distances ) > 1E29 :
                raise RuntimeError( "GetNeighbors() Library is too small to " +\
                                    "resolve " + str( k_NN ) + " k_NN "   +\
                                    "neighbors." )
        
            # Check for ties.  JP: haven't found any so far...
            if len( k_NN_neighbors ) != len( unique( k_NN_neighbors ) ) :
                raise RuntimeError( "GetNeighbors() Degenerate neighbors" )
        
        neighbors[ row, ] = k_NN_neighbors
        distances[ row, ] = k_NN_distances

        row = row + 1
    
    if args.Debug :
        print( 'GetNeighbors()  neighbors' )
        print( neighbors[ 0:5, ] )

    return ( neighbors, distances )
    
#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    CCM()
