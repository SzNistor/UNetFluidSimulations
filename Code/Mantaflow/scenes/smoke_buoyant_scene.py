#******************************************************************************
#
# MantaGen
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0 
# http://www.apache.org/licenses/LICENSE-2.0
#
#******************************************************************************
import random

from manta import *

import numpy
from random import randint
from scenes.scene import Scene
from scenes.volumes import *
from scenes.functions import *
from util.logger import *


def instantiate_scene(**kwargs): # instantiate independent of name , TODO replace?
    info(kwargs)
    return SmokeBuoyantScene(**kwargs) 


class SmokeBuoyantScene(Scene):
    #----------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super(SmokeBuoyantScene,self).__init__(**kwargs)
        # optionally, init more grids etc.

        self.max_iter_fac = 2
        self.accuracy     = 5e-4
        self.max_source_count   = int(kwargs.get("max_source_count", 1))
        self.velocity_scale     = float(kwargs.get("velocity_scale", self.resolution.y * 0.05))
        self.use_inflow_sources = kwargs.get("use_inflow_sources", "True") == "True"
        self.open_bound         = kwargs.get("use_open_bound", "True") == "True"
        self.sources            = []
        self.source_strengths   = []
        
        # ADDED BY Szilard Nistor
        # Initiate scene with additional vorticity
        self.vorticity_strength = float(kwargs.get("vorticity", 0.))

        # smoke sims need to track the density
        self.density = self.solver.create(RealGrid, name="Density")
        
        # CHANGED BY Szilard Nistor
        # Set the noise to 0, so that the smoke source remains constant
        noise = self.solver.create(NoiseField, loadFromFile=True)
        noise.posScale = vec3(40) * 0 #* numpy.random.uniform(low=0.25, high=1.)
        noise.posOffset = random_vec3s(vmin=0.0) * 0 #* 100.
        noise.clamp = True
        noise.clampNeg = 0
        noise.clampPos = 1.
        noise.valOffset = 1 #0.15
        noise.timeAnim  = 0.4 * numpy.random.uniform(low=0.2, high=1.)
        self.noise = noise

        info("SmokeBuoyantScene initialized")

    #----------------------------------------------------------------------------------
    def set_velocity(self, volume, velocity):
        if self.dimension == 2:
            velocity.z = 0.0
        volume.applyToGrid(solver=self.solver, grid=self.vel, value=velocity)

    #----------------------------------------------------------------------------------
    # sources used as smoke inflow in the following
    def add_source(self, volume):
        shape = volume.shape(self.solver)
        self.sources.append(shape)
        self.source_strengths.append(numpy.random.uniform(low=0.1, high=1.))

    # ----------------------------------------------------------------------------------
    # ADDED BY Szilard Nistor
    # Add obstacles to the scene and register them on the grid
    def add_obstacle(self, volume):
        vol_ls = volume.computeLevelset(self.solver)
        volume.applyToGrid(solver=self.solver, grid=self.flags, value=2)
        self.phi_obs.join(vol_ls)


    #----------------------------------------------------------------------------------
    def _create_scene(self):
        super(SmokeBuoyantScene, self)._create_scene()

        self.sources = []
        self.source_strengths = []
        self.density.setConst(0) 
        self.vel.setConst(vec3(0)) 
        is3d = (self.dimension > 2)

        self.flags.initDomain(boundaryWidth=self.boundary)
        self.flags.fillGrid()

        setOpenBound(self.flags, self.boundary, 'xXyYzZ', CellType_TypeOutflow|CellType_TypeEmpty)

        # formerly initialize_smoke_scene(scene):
        source_count = randint(1, self.max_source_count)
        for i in range(source_count):
            volume = random_box(center_min=[0.1, 0.05, 1], center_max=[0.9, 0.4, 1], size_min=[0.01, 0.01, 0.01], size_max=[0.2, 0.2, 0.2], is3d=is3d)
            self.add_source(volume)
            src, sstr = self.sources[-1], self.source_strengths[-1]
            
            # ADDED BY Szilard Nistor
            # Add up to four random obstacels to the upper part of the scene
            obstacles_nr = randint(0, 4)
            for i in range(0, obstacles_nr):
                obstacle = SphereVolume(center=random_vec3([0.1+(0.8/obstacles_nr)*i, 0.4, 1], [0.1+(0.8/obstacles_nr)*(i+1), 1, 1]), radius=random.uniform(0.01, 0.15))
                self.add_obstacle(obstacle)

            densityInflow(flags=self.flags, density=self.density, noise=self.noise, shape=src, scale=2.0*sstr, sigma=0.5)

        if self.show_gui:
            # central view is more interesting for smoke
            self._gui.setPlane(self.resolution.z // 2)

        info("SmokeBuoyantScene created with {} sources".format(len(self.sources)))

    #==================================================================================
    # SIMULATION
    #----------------------------------------------------------------------------------

    def _compute_simulation_step(self):
        # Note - sources are turned off earlier, the more there are in the scene
        for i in range(len(self.sources)):
            if self.use_inflow_sources:
                src, sstr = self.sources[i], self.source_strengths[i]
                densityInflow(flags=self.flags, density=self.density, noise=self.noise, shape=src, scale=2.0*sstr, sigma=0.5)

        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.density, order=2, clampMode=2)
        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.vel    , order=2, clampMode=2)

        # CHANGED BY Szilard Nistor
        # Add the additional vorticity strength to the original value of 0.1
        vorticityConfinement(vel=self.vel, flags=self.flags, strength=0.1 + self.vorticity_strength)
        addBuoyancy(density=self.density, vel=self.vel, gravity=0.2*self.gravity, flags=self.flags)


        setWallBcs(flags=self.flags, vel=self.vel, phiObs=self.phi_obs)


        solvePressure(flags=self.flags, vel=self.vel, pressure=self.pressure, cgMaxIterFac=self.max_iter_fac, cgAccuracy=self.accuracy)
