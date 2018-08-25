#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from EDM      import EmbedData

#----------------------------------------------------------------------------
# __main__
#----------------------------------------------------------------------------
def Embed():
    '''
    Time-delay embedd data vector(s) from args.inputFile into 
    args.Dimensions at multiples of args.tau.  Note that if the 
    -f --forwardTau option is specified, then the embedding is 
    x(t) + τ instead of x(t) - τ.

    The -e (embedColumns) option specifies the zero-offset column
    numbers or column names to embed from args.inputFile.

    Writes a .csv file with header [Time, Dim_1, Dim_2...] if -o specified.

    Note: The output .csv file will have fewer rows (observations)
    than the input data by args.Dimension - 1 (E-1). 
    '''
    
    args = ParseCmdLine()

    embedding, header, target = EmbedData( args )

    if args.Debug:
        print( "Embed() " + ' '.join( args.embedColumns ) +\
               " from " + args.inputFile +\
               " E=" + str( args.E ) + " " +\
               str( embedding.shape[0] ) + " rows,  " +\
               str( embedding.shape[1] ) + " columns." )
        
        print( header )
        print( embedding[ 0:3, : ] )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def TestEmbed( N_row = 10, E = 4, tau = 1 ) :
    
    import numpy as np
    m     = np.zeros( ( N_row, E+1 ) )
    data  = np.zeros( ( N_row, 2 ) )
    data[ :, 0 ] = np.arange( 1, N_row + 1 )
    data[ :, 1 ] = data[ :, 0 ] + 10
    
    m[ :, 0 ] = data[ :, 0 ] # time vector
    m[ :, 1 ] = data[ :, 1 ] # original data vector from column 1

    print( str( m ) )
    print( "-------------------------------" )

    for j in range( 2, E + 1 ) :
        delta_row = ( j - 1 ) * tau
        m[ delta_row : N_row : 1, j ] = data[ 0 : N_row - delta_row : 1, 1 ]

    del_row = (E-1) * tau
    m = np.delete( m, np.s_[ 0 : del_row : 1 ], 0 )

    print( str( m ) )
    
#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    Embed()
