#! /usr/bin/env python3

# Python distribution modules
from collections     import OrderedDict
from itertools       import combinations
from multiprocessing import Pool

# Community modules
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.dates import num2date

# Local modules
from ArgParse import ParseCmdLine
from EDM      import EmbedData, ReadEmbeddedData, Prediction, \
                     ComputeError, nCol, nRow

#----------------------------------------------------------------------------
# __main__
#----------------------------------------------------------------------------
def Multiview():
    '''
    Data input requires -c (columns) to specify timeseries columns
    in inputFile (-i) that will be embedded by EmbedData(), and the 
    -r (target) specifying the data target column in inputFile.

    args.E represents the number of variables to combine for each
    assessment, as well as the number of time delays to create in 
    EmbedData() for each variable. 

    Prediction() with Simplex sets k_NN equal to E+1 if -k not specified.

    --
    Ye H., and G. Sugihara, 2016. Information leverage in interconnected 
    ecosystems: Overcoming the curse of dimensionality. 
    Science 353:922–925.
    '''

    args = ParseCmdLine()

    if not len( args.columns ) :
        raise RuntimeError( 'Multiview() requires -c to specify data.' )
    if not args.target :
        raise RuntimeError( 'Multiview() requires -r to specify target.' )
    if args.E < 0 :
        raise RuntimeError( 'Multiview() E is required.' )

    # Save args.plot flag, and disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    # Save args.outputFile and reset so Prediction() does not write 
    outputFile      = args.outputFile
    args.outputFile = None

    # Embed data from inputFile
    embedding, colNames, target = EmbedData( args )

    # Combinations of possible embedding variables (columns), E at-a-time
    # Column 0 is time. Coerce the iterable into a list of E-tuples
    nVar   = len( args.columns )
    combos = list( combinations( range( 1, nVar * args.E + 1 ), args.E ) )

    # Require that each embedding has at least one coordinate with
    # observed data (zero time lag). This corresponds to combo tuples
    # with modulo E == 1.
    # Note: this only works if the data (unlagged) are in columns
    # 1, 1 + E, 1 + 2E, ... which is consistent with EmbedData() output.
    combo_i = []
    for i in range( len( combos ) ) :
        c = combos[i] # a tuple of combination indices
        for x in c:
            if x % args.E == 1:
                combo_i.append( i )
                break

    combos = [ combos[i] for i in combo_i ]

    if not args.multiview :
        # Ye & Sugihara suggest sqrt( m ) as the number of embeddings to avg
        args.multiview = max( 2, int( np.sqrt( len( combos ) ) ) )
        
        print( 'Multiview() Set view sample size to ' + str( args.multiview ))
        
    #---------------------------------------------------------------
    # Evaluate variable combinations.
    # Note that this is done within the library itself (in-sample).
    # Save a copy of the specified prediction observations.
    prediction = args.prediction

    # Override the args.prediction for in-sample forecast skill evaluation
    args.prediction = args.library
    
    # Process pool to evaluate combos
    pool = Pool()

    # Iterable list of arguments for EvalLib()
    argList = []
    for combo in combos :
        argList.append( ( args, combo, embedding, colNames, target ) )
    
    # Submit EvalLib jobs to the process pool
    results = pool.map( EvalLib, argList )
    
    # Dict to hold combos : rho pairs from EvalLib() tuple
    Combo_rho = {}

    for result in results :
        if result == None:
            continue
        Combo_rho[ result[ 0 ] ] = result[ 1 ]

    #---------------------------------------------------------------
    # Rank the in-sample forecasts, zip returns an iterator of 1-tuples
    rho_sort, combo_sort = zip( *sorted( zip( Combo_rho.values(),
                                              Combo_rho.keys() ),
                                         reverse = True ) )
    
    if args.Debug:
        print( "Multiview()  In sample sorted embeddings:" )
        print( 'Columns         ρ' )
        for i in range( min( args.multiview, len( combo_sort ) ) ):
            print(str( combo_sort[i] ) + "    " + str( round( rho_sort[i],4)))
    
    #---------------------------------------------------------------
    # Perform predictions with the top args.multiview embeddings
    # Reset the user specified prediction vector
    args.prediction = prediction
    
    argList.clear() # Iterable list of arguments for EvalPred()

    # Take the top args.multiview combos
    for combo in combo_sort[ 0: args.multiview ] :
        argList.append( ( args, combo, embedding, colNames, target ) )
    
    # Submit EvalPred jobs to the process pool
    results = pool.map( EvalPred, argList )
    
    Results = OrderedDict() # Dictionary of dictionaries results each combo

    for result in results :
        if result == None:
            continue 
        Results[ result[ 0 ] ] = result[ 1 ]

    # Console output
    print( "Multiview()  Prediction Embeddings:" )
    print( "Columns       Names                       ρ       mae   rmse" )
    for key in Results.keys() :
        result = Results[ key ]
        print( str( key ) + "   " + ' '.join( result[ 'names' ] ) +\
               "  " + str( round( result[ 'rho'  ], 4 ) ) +\
               "  " + str( round( result[ 'mae'  ], 4 ) ) +\
               "  " + str( round( result[ 'rmse' ], 4 ) ) )

    #----------------------------------------------------------
    # Compute Multiview averaged prediction
    # The output item of Results dictionary is a matrix with three
    # columns [ Time, Data, Prediction_t() ]
    # Collect the Predictions into a single matrix
    aresult = Results[ combo_sort[0] ]
    nrows   = nRow( aresult['output'] )
    time    = aresult['output'][:,0]
    data    = aresult['output'][:,1]
    
    M = np.zeros( ( nrows, len( Results ) ) )

    col_i = 0
    for result in Results.values() :
        output = result[ 'output' ]
        M[ :, col_i ] = output[ :, 2 ] # Prediction is in col j=2
        col_i = col_i + 1

    prediction    = np.mean( M, axis = 1 )
    multiview_out = np.column_stack( ( time, data, prediction ) )

    # Write output
    if outputFile:
        header = 'Time,Data,Prediction_t(+{0:d})'.format( args.Tp )
        np.savetxt( args.path + outputFile, multiview_out, fmt = '%.4f',
                    delimiter = ',', header = header, comments = '' )

    # Estimate correlation coefficient on observed : predicted data
    rho, r, rmse, mae = ComputeError( data, prediction )

    print( ("Multiview()  ρ {0:5.3f}  r {1:5.3f}  RMSE {2:5.3f}  "
            "MAE {3:5.3f}").format( rho, r, rmse, mae ) )
    
    #----------------------------------------------------------
    if showPlot:
        
        Time = multiview_out[ :, 0 ] # Required to be first (j=0) column

        if args.plotDate :
            Time = num2date( Time )
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        
        ax.plot( Time, multiview_out[ :, 1 ],
                 label = 'Observations',
                 color='blue', linewidth = 2 )
        
        ax.plot( Time, multiview_out[ :, 2 ],
                 label = 'Predictions_t(+{0:d})'.format( args.Tp ),
                 color='red', linewidth = 2 )

        if args.verbose :  # Plot all projections
            for col in range( nCol( M ) ) :
                ax.plot( multiview_out[ :, 0 ], M[ :, col ],
                         label = combo_sort[col], linewidth = 2 )
        
        ax.legend()
        ax.set( xlabel = args.plotXLabel,
                ylabel = args.plotYLabel,
                title  = "Multiview  " + args.inputFile +\
                         ' Tp=' + str( args.Tp ) +\
                         ' E='  + str( args.E ) + r' $\rho$=' +\
                str( round( rho, 2 ) ) )
        plt.show()

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def EvalLib( argsList ) :
    '''
    Function for multiprocessing of combo evaluation within library
    argsList : ( args, combo, embedding, colNames, target )
    '''
    args      = argsList[ 0 ]
    combo     = argsList[ 1 ]
    embedding = argsList[ 2 ]
    colNames  = argsList[ 3 ]
    target    = argsList[ 4 ]
    
    # Extract the variable combination
    # Note that we prepend the time column (0,) as Prediction() requires
    embed = np.take( embedding, (0,) + combo, axis = 1 )
    
    # Evaluate prediction skill
    rho, rmse, mae, header, output, smap_output = Prediction( embed,
                                                              colNames,
                                                              target, args )
    
    return tuple( ( combo, round( rho, 5 ) ) )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def EvalPred( argsList ) :
    '''
    Function for multiprocessing of combo evaluation outside library
    argsList : ( args, combo, embedding, colNames, target )
    '''
    args      = argsList[ 0 ]
    combo     = argsList[ 1 ]
    embedding = argsList[ 2 ]
    colNames  = argsList[ 3 ]
    target    = argsList[ 4 ]
        
    # Extract the variable combination
    # Note that we prepend the time column (0,) as Prediction() requires
    embed = np.take( embedding, (0,) + combo, axis = 1 )
    Names = [ colNames[i] for i in combo ]
    
    # Evaluate prediction skill
    rho, rmse, mae, header, output, smap_output = Prediction( embed,
                                                              colNames,
                                                              target, args )
    
    Result = { 'names' : Names,  'rho'    : rho,
               'rmse'  : rmse,   'mae'    : mae,
               'header': header, 'output' : output }

    return tuple( ( combo, Result ) )

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    Multiview()
