#!/usr/bin/gnuplot
set output "eccentricity.pdf"
set terminal pdf enhanced  dashed size 3.5in, 2.5in 

set ylabel "number of planets"
set xlabel "eccentricity"
set key bottom right
set xrange [0:.5]
set ytics 200
plot  "eccentricity.integrated.out_stochastic_1e-5" u 1:0 w l lc -1  t "{/Symbol t}_a=10^{4} years, {/Symbol a}=10^{-5}",\
 "eccentricity.integrated.out_stochastic_1e-6" u 1:0 w l lc -1  t "{/Symbol t}_a=10^{4} years, {/Symbol a}=10^{-6}", \
 "eccentricity.integrated.out_migration_1e3" u 1:0 w l lc 8   t "{/Symbol t}_a=10^{3} years", \
 "eccentricity.integrated.out_migration_1e4" u 1:0 w l lc 8  t "{/Symbol t}_a=10^{4} years"
