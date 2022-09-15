# from lmm import *


# ctp = [ 0.25 * ( i + 1 ) for i in range( 10 )]
# r0  = RateStructure( 0.0, [ 0.005, 0.01, 0.016, 0.02, 0.024, 0.027, 0.03, 0.033, 0.035, 0.037 ] )
# dt  = [ 0.25 for i in range( 10 ) ]
# ind = 5

# tarVol = [0.38,0.385,0.396666667,0.405,0.412,0.411666667,0.408571429,0.405,0.398888889,0.392 ]

# tarCor = [ (0.25, 1.0, 0.3 ), ( 0.25, 1.25, 0.32 ), (0.25, 1.5, 0.33), (0.5, 1.0, 0.31), (0.5, 1.5, 0.34 ) ]

# vola = vol2( [ 0.2 for i in range( 10 ) ] )
# volb = vol6( 0.2, 0.3, 0.4, 0.5 )
# volc = vol7( 0.2, 0.3, 0.4, 0.5, [ 0.2 for i in range( 11 ) ] )

# cora = Corr1( 0.9 )
# corb = Corr2( 0.9, 0.9 )


# import random


# a = vol2( [ random.random() for i in range( 10 ) ] )
# print(a.calibrate([0.38,0.385,0.396666667,0.405,0.412,0.411666667,0.408571429,0.405,0.398888889,0.392 ], [0.25*i for i in range( 1, 11 )], \
#             [0.25 for i in range( 10 ) ] ))


# testLMM = IterPredCorrect( r0, vola, cora, ctp, dt, ind, 0.25 )

# testLMM.calibrate( tarVol, tarCor )


# print( testLMM.getDrift( r0 ) )



# aRes = testLMM.simulate( 1.5 )




# bisect.bisect_left( ctp, 0.25 + 0.25 )



# print( aRes[ 2 ])




# testLMM.getDrift( aRes[ 2 ] )




