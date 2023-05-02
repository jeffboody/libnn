# gnuplot
# load "output.plot"

# style
set style line 1 lt rgb "#FF0000" lw 3 pt 6
set style line 2 lt rgb "#00FF00" lw 3 pt 6
set style line 3 lt rgb "#0000FF" lw 3 pt 6

# plot data
set multiplot layout 2, 2;
plot "output-0.dat" using 1:3 with linespoints ls 1 title 'Y',\
     "output-0.dat" using 1:1 with linespoints ls 2 title 'X',\
     "output-0.dat" using 1:2 with linespoints ls 3 title 'Yt'
plot "output-3.dat" using 1:3 with linespoints ls 1 title 'Y',\
     "output-3.dat" using 1:1 with linespoints ls 2 title 'X',\
     "output-3.dat" using 1:2 with linespoints ls 3 title 'Yt'
plot "output-6.dat" using 1:3 with linespoints ls 1 title 'Y',\
     "output-6.dat" using 1:1 with linespoints ls 2 title 'X',\
     "output-6.dat" using 1:2 with linespoints ls 3 title 'Yt'
plot "output-9.dat" using 1:3 with linespoints ls 1 title 'Y',\
     "output-9.dat" using 1:1 with linespoints ls 2 title 'X',\
     "output-9.dat" using 1:2 with linespoints ls 3 title 'Yt'
unset multiplot
