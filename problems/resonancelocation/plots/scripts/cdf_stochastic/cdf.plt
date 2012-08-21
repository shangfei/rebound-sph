#!/usr/bin/gnuplot
set output "cdf_stochastic.pdf"
set terminal pdf enhanced  dashed size 3.5in, 2.5in 

set ylabel "number of pairs"
set xlabel "period ratio"
set xrange [1:9]
set logscale x
set yrange [0:*]
set grid x
set xtics (1, 1.33, 1.5, 2.,3.,4.,5.,6.,7.,8.,9.)
set cblabel "total planet mass m_1 + m_2 [M_{jup}]"
set key top left
set cbrange [0.:.1]
set multiplot
plot "cdf.kepler" u 1:0 w l t "Observed KOI planets", \
"cdf.integrated.out_stochastic_1e-6" u 1:0 w l t "Migration {/Symbol t}_a=10^{4} years, {/Symbol a}=10^{-6}"  ls 4 lc -1, \
"cdf.integrated.out_stochastic_1e-5" u 1:0 w l t "Migration {/Symbol t}_a=10^{4} years, {/Symbol a}=10^{-5}"  ls 2 lc -1

set style rect fc lt -1 fs solid 0.15 noborder
set object 1 rect from screen 0.6,0.2 to screen 0.9,0.56 fc rgb "white" behind 
set autoscale fix
set xrange [1.9:2.1]
set yrange [180:310]
set ytics 50
set lmargin at screen 0.6
set rmargin at screen 0.9
set tmargin at screen 0.56
set bmargin at screen 0.2
unset xlabel
unset ylabel
set xtics (1.9, 2.,2.1)
plot "cdf.kepler" u 1:0 w l notit, \
"cdf.integrated.out_stochastic_1e-6" u 1:0 w l notit ls 4 lc -1, \
"cdf.integrated.out_stochastic_1e-5" u 1:0 w l notit  ls 2 lc -1
