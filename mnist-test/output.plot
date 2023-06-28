# gnuplot
# load "output.plot"

# style
set style line 1 lt rgb "#FF0000" lw 3 pt 6
set style line 2 lt rgb "#00FF00" lw 3 pt 6
set style line 3 lt rgb "#0000FF" lw 3 pt 6

# plot data
plot "data/plot.dat" using 2:5 with linespoints ls 1 title 'max', \
     "data/plot.dat" using 2:4 with linespoints ls 2 title 'min', \
     "data/plot.dat" using 2:3 with linespoints ls 3 title 'avg'
