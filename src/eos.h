/**
 * @file 	hydro.h
 * @brief 	Equation of state 
 * @author 	Shangfei Liu <shangfei.liu@gmail.com>
 *
 * @section LICENSE
 * Copyright (c) 2011 Hanno Rein, Shangfei Liu
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
#ifndef _EOS_H
#define _EOS_H
struct reb_simulation;

/**
  * The function loops over all ghostboxs and calls calculate_forces_for_particle() to sum up the forces on each particle.
  * Calculate all the gravitational acceleration for all particles.
  * Different methods implement this function in a different way.
  */
  void reb_eos_init(struct reb_simulation* const r);
  
  void reb_eos (const struct reb_simulation* const r, const int pt);

  void reb_calculate_internal_energy_for_sph_particle(struct reb_simulation* r, int pt);

#endif
