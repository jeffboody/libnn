# gnuplot
# load "output.plot"

# style
set style line 1 lt rgb "#FF0000" lw 3 pt 6
set style line 2 lt rgb "#00FF00" lw 3 pt 6
set style line 3 lt rgb "#0000FF" lw 3 pt 6
set style line 4 lt rgb "#FF00FF" lw 3 pt 6
set style line 5 lt rgb "#FFFF00" lw 3 pt 6
set style line 6 lt rgb "#00FFFF" lw 3 pt 6

# plot data
plot "data/plot.dat" using 2:5 with linespoints ls 1 title 'D:max', \
     "data/plot.dat" using 2:4 with linespoints ls 2 title 'D:min', \
     "data/plot.dat" using 2:3 with linespoints ls 3 title 'D:avg', \
     "data/plot.dat" using 2:8 with linespoints ls 4 title 'G:max', \
     "data/plot.dat" using 2:7 with linespoints ls 5 title 'G:min', \
     "data/plot.dat" using 2:6 with linespoints ls 6 title 'G:avg'
