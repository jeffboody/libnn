# gnuplot
# load "output.plot"

# style
set style line 1  lt rgb "#FF0000" lw 3 pt 6
set style line 2  lt rgb "#00FF00" lw 3 pt 6
set style line 3  lt rgb "#0000FF" lw 3 pt 6
set style line 4  lt rgb "#FF00FF" lw 3 pt 6
set style line 5  lt rgb "#FFFF00" lw 3 pt 6
set style line 6  lt rgb "#00FFFF" lw 3 pt 6
set style line 7  lt rgb "#404040" lw 3 pt 6
set style line 8  lt rgb "#C0C0C0" lw 3 pt 6
set style line 9  lt rgb "#808080" lw 3 pt 6
set style line 10 lt rgb "#000000" lw 3 pt 6

# plot data
plot "data/plot.dat" using 2:5  with linespoints ls 7  title 'Loss Max', \
     "data/plot.dat" using 2:4  with linespoints ls 8  title 'Loss Min', \
     "data/plot.dat" using 2:3  with linespoints ls 9  title 'Loss Avg', \
     "data/plot.dat" using 2:8  with linespoints ls 1  title 'G Max',    \
     "data/plot.dat" using 2:7  with linespoints ls 2  title 'G Min',    \
     "data/plot.dat" using 2:6  with linespoints ls 3  title 'G Avg',    \
     "data/plot.dat" using 2:11 with linespoints ls 4  title 'D Max',    \
     "data/plot.dat" using 2:10 with linespoints ls 5  title 'D Min',    \
     "data/plot.dat" using 2:9  with linespoints ls 6  title 'D Avg',    \
     "data/plot.dat" using 2:12 with linespoints ls 10 title 'Blend Factor'
