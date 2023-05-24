# gnuplot
# load "output.plot"

# style
set style line 1 lt rgb "#FF0000" lw 3 pt 6

# plot data
plot "data/plot.dat" using 1:2 with linespoints ls 1 title 'loss'
