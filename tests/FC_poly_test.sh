# This bash script runs the python script FC_poly.py in the 
#  current polytrope directory and in another polytrope directory
#  in order to see the difference between the two runs for a given Ra and Pr.  
#  tests/FC_poly_test.py then outputs some plots of the differences between some
#  important scalars and profiles between the two runs to determine if changes
#  in the backend have affected user experience.
#
# Usage:
#   In the master directory of the polytrope/ repo, type:
#           bash tests/FC_poly_test.sh > tests/FC_poly_test.out
#
# I recommend funnelling the output to a file (as in usage example above) so
#   that you can directly compare the output of the two runs once the test
#   is finished and you've exited the terminal.

##############################BASICS#################################################
FILENAME=FC_poly.py
RAYLEIGH=1e5    #Must be input in the format that it will be output [see OUTDIR1]
PRANDTL=1
RUNTIME=0.1 #~6 min runs, units in hours
RUNTIME_SEC=$(printf "%.0f" $(echo $RUNTIME*3600 | bc))

# MPI specification
NUMPROCS=16
MPICOMMAND=mpirun  #mpiexec_mpt on pleiades
MPI=$(printf "%s %s %d " $MPICOMMAND "-n" $NUMPROCS )

# DIR1 is the current directory we're testing
# DIR2 should be set to the TOTAL path of the other polytrope/ we care about
DIR1=$PWD
DIR2=$HOME/code_comp/polytrope-merging/  #$PWD/../polytrope-old/

# The FC_poly.py files will output to OUTDIR1 and OUTDIR2, respectively 
#       (if FC_poly outputting is changed  in this commit, you need to 
#       change one or the other.  If not, they should match). 
# Resultant test plots will appear in TEST_OUTDIR
OUTDIR1=FC_poly_constNu_constKappa_nrhocz3.5_Ra$RAYLEIGH\_Pr$PRANDTL\_eps1e-4
OUTDIR2=FC_poly_constNu_constKappa_nrhocz3.5_Ra$RAYLEIGH\_Pr$PRANDTL\_eps1e-4
TEST_OUTDIR=$DIR1/tests/test_FC_poly_basic/
######################################################################################


################################RUNNING###############################################


# Run for a few min each, merge (merge happens in FC_poly.py script)
# The time checks and if statements are included in case you kill the script early so 
#   that it doesn't try to keep running other python things.
cd $DIR1
time1=$(date +%s)

#Clear old output directories of the same name
rm -rf $DIR1/$OUTDIR1
rm -rf $DIR2/$OUTDIR2

#Run1
$MPI python3 $FILENAME \
    --run_time=$RUNTIME  \
    --Rayleigh=$RAYLEIGH \
    --Prandtl=$PRANDTL

time2=$(date +%s)
dt=`expr $time2 - $time1`
if [ $dt -lt $RUNTIME_SEC ]
then
    echo Something caused the run to crash after $dt seconds, exiting
    exit 1
fi



time1=$(date +%s)
cd $DIR2

#Run2
$MPI python3 $FILENAME \
    --run_time=$RUNTIME  \
    --Rayleigh=$RAYLEIGH \
    --Prandtl=$PRANDTL

time2=$(date +%s)
dt=`expr $time2 - $time1`
if [ $dt -lt $RUNTIME_SEC ]
then
    echo Something caused the run to crash after $dt seconds, exiting
    exit 1
fi



######################################################################################



#######################################COMPARING######################################
#Once the runs are done, make some plots comparing them
cd $DIR1
python3 tests/FC_poly_test.py \
        --outdir1=$DIR1/$OUTDIR1/ \
        --outdir2=$DIR2/$OUTDIR2/ \
        --plot_dir=$TEST_OUTDIR
