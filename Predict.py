#! /usr/bin/env python3

from ArgParse import ParseCmdLine
from EDM      import EmbedData, ReadEmbeddedData, Prediction

#----------------------------------------------------------------------------
# __main__
#----------------------------------------------------------------------------
def Predict():
    '''
    Wrapper for Precition() in EMD.py, performs:

      Simplex projection of observational data (Sugihara, 1990), or
      SMap    projection of observational data (Sugihara, 1994).

    There are two options for data input. One is to use the -c (columns)
    argument so that the -i (inputFile) will be considered a timeseries with 
    embedding performed by EmbedData().  The other is to not use -e where
    the -i (inputFile) specifies a .csv file with an embedding or
    multivariable data frame. This will be read by ReadEmbeddedData().

    If ReadEmbeddedData() is used (-e not specified) then input data 
    consist of a .csv file formatted as:
       [ Time, Dim_1, Dim_2, ... ] 
    where Dim_1 is observed data, Dim_2 data offset by τ, Dim_3 by 2τ...
    The user can specify the desired embedding dimension E, which
    can be less than the total number of columns in the inputFile. 
    The first E + 1 columns (Time, D1, D2, ... D_E) will be returned.

    Alternatively, the data can be a .csv file with multiple simultaneous
    observations or delay embeddings (columns) where the columns to 
    embed and target to project are specified with the -c (columns)
    and -r (target) options. In both cases 'time' is required in column 0. 
 
    Embedding can be done with EmbedData() via the wrapper Embed.py. 
    Note: The embedded data .csv file will have fewer rows (observations)
    than the data input to EmbedData() by E - 1. 
    '''
    
    args = ParseCmdLine()

    if not args.embedded :
        # args.inputFile are timeseries data to be embedded by EmbedData
        embedding, colNames, target = EmbedData( args )
    else :
        # The args.inputFile is an embedding or multivariable data frame.
        # ReadEmbeddedData() sets args.E to the number of columns
        # if the -c (columns) and -t (target) options are used.
        embedding, colNames, target = ReadEmbeddedData( args )

    rho, rmse, mae, header, output = Prediction( embedding, colNames,
                                                 target, args )
    

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    Predict()
