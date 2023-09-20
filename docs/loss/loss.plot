# gnuplot
# load "loss.plot"

# plot data
set multiplot layout 1,3
set pm3d
set hidden3d
splot "mse.dat" matrix
splot "mae.dat" matrix
splot "bce.dat" matrix
unset multiplot
