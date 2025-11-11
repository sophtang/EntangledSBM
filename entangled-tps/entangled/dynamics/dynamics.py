# We refer to https://github.com/LarsHoldijk/SOCTransitionPaths/blob/master/potentials/alanine_md.py
# for applying neural network bias force to OpenMM

import openmm as mm
from openmm import app
import openmm.unit as unit
from openmmtools.integrators import VVVRIntegrator

from .base import BaseDynamics


class Aldp(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField("amber99sbildn.xml")
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Chignolin(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Trpcage(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Bba(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Bbl(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Poly(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force
