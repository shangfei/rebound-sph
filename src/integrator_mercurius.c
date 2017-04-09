/**
 * @file    integrator_mercurius.c
 * @brief   MERCURIUS, an improved version of John Chambers' MERCURY algorithm
 * @author  Hanno Rein
 * 
 * @section LICENSE
 * Copyright (c) 2017 Hanno Rein 
 *
 * This file is part of rebound.
 *
 * rebound is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rebound is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "rebound.h"
#include "output.h"
#include "integrator_mercurius.h"
#include "integrator_ias15.h"
#include "integrator_whfast.h"
#include "integrator_whfasthelio.h"


void reb_integrator_mercurius_part1(struct reb_simulation* r){
	const double dt = r->dt;
    r->gravity_ignore_terms = 0;
	r->t+=dt/2.;
}

void reb_integrator_mercurius_part2(struct reb_simulation* r){
	const double dt = r->dt;
	r->t+=dt/2.;
}

void reb_integrator_mercurius_synchronize(struct reb_simulation* r){
}

void reb_integrator_mercurius_reset(struct reb_simulation* r){
    r->ri_mercurius.mode = 0;
}

