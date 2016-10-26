"convert libsvm file to vw format"
"skip malformed lines"

import sys
import argparse

parser = argparse.ArgumentParser( description = "convert libsvm file to vw format, skip malformed lines" )
parser.add_argument( "input_file", help = "path to csv input file" )
parser.add_argument( "output_file", help = "path to output file" )
parser.add_argument( "-z", "--convert_zeros", help = "convert labels for binary classification from 0 to -1", 
	action = 'store_true', default = False )
parser.add_argument("-v", "--verbose", help = "will write line counts to standard out",
					action = "store_true", default = False )

args = parser.parse_args()

i = open( args.input_file, 'r' )
o = open( args.output_file, 'w' )

count = 0
for line in i:
	try:
		y, x = line.split( " ", 1 )
	# ValueError: need more than 1 value to unpack
	except ValueError:
		print("line with ValueError (skipping):")
		print(line)
		continue

	if args.convert_zeros and float( y ) != 1.0:
		y = '-1'
		
	new_line = y + " |n " + x
	o.write( new_line )
	
	count += 1
	if args.verbose and count % 100000 == 0:
		print(count)
