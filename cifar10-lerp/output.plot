# gnuplot
# load "output.plot"

# style
set style line 1 lt rgb "#FF0000" lw 3 pt 6
set style line 2 lt rgb "#00FF00" lw 3 pt 6
set style line 3 lt rgb "#0000FF" lw 3 pt 6
set style line 4 lt rgb "#FF00FF" lw 3 pt 6
set style line 5 lt rgb "#FFFF00" lw 3 pt 6
set style line 6 lt rgb "#00FFFF" lw 3 pt 6
set style line 7 lt rgb "#404040" lw 3 pt 6
set style line 8 lt rgb "#C0C0C0" lw 3 pt 6
set style line 9 lt rgb "#808080" lw 3 pt 6

# plot data
plot "data/plot.dat" using 2:5  with linespoints ls 1 title 'max1', \
     "data/plot.dat" using 2:4  with linespoints ls 2 title 'min1', \
     "data/plot.dat" using 2:3  with linespoints ls 3 title 'avg1', \
     "data/plot.dat" using 2:8  with linespoints ls 4 title 'max2', \
     "data/plot.dat" using 2:7  with linespoints ls 5 title 'min2', \
     "data/plot.dat" using 2:6  with linespoints ls 6 title 'avg2', \
     "data/plot.dat" using 2:11 with linespoints ls 7 title 'max3', \
     "data/plot.dat" using 2:10 with linespoints ls 8 title 'min3', \
     "data/plot.dat" using 2:9  with linespoints ls 9 title 'avg3'
