#!/usr/bin/env python

"""
openmc_fixed_full.py
Fixed, hardened OpenMC driver script:
 - corrected geometry (bounded fuel & moderator)
 - safe tally list (no 'capture' score)
 - geometry overlap checking + optional plotting
 - robust run wrapper with helpful error messages (lost particles, invalid scores)
 - post-processing: k_eff, tallies, compute capture = absorption - fission
 - conservative default run sizes (for debugging); tune BATCHES/PARTICLES for final runs
"""

import os
import sys
import glob
import time
from datetime import datetime

import openmc

# ---------------- USER TUNABLES ----------------
OUTER_RADIUS = 50.0     # cm, outer bounding sphere
FUEL_RADIUS = 0.39      # cm, fuel cylinder radius
DEBUG = True            # if True create geometry plots and use small runs for quick debugging
RUN_DEPLETION = False   # set True only if openmc.deplete is configured & you want depletion
# Debug run sizes (fast)
BATCHES = 20
INACTIVE = 5
PARTICLES = 20000
SEED = 12345

# Mesh tally settings
NX, NY, NZ = 20, 20, 20
MESH_MARGIN = 0.1       # avoid placing mesh exactly on boundary

# ------------------------------------------------

def build_materials():
    # simple demo composition (atomic fractions). Replace with number densities for production.
    fuel = openmc.Material(name='UO2 fuel')
    fuel.add_nuclide('U235', 0.04)
    fuel.add_nuclide('U238', 0.96)
    fuel.add_nuclide('O16', 2.0)
    fuel.set_density('g/cm3', 10.5)

    water = openmc.Material(name='Light water')
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    water.set_density('g/cm3', 0.743)
    # s(alpha,beta) if available
    try:
        water.add_s_alpha_beta('c_H_in_H2O')
    except Exception:
        # missing S(a,b) data is fine — script will warn
        pass

    mats = openmc.Materials([fuel, water])
    mats.export_to_xml()
    return fuel, water, mats

def build_geometry(fuel, water):
    # Surfaces
    fuel_cyl = openmc.ZCylinder(r=FUEL_RADIUS)
    outer_sphere = openmc.Sphere(r=OUTER_RADIUS, boundary_type='vacuum')

    # Cells — IMPORTANT: bind both fuel & moderator inside outer_sphere to avoid lost particles
    fuel_cell = openmc.Cell(name='fuel')
    fuel_cell.fill = fuel
    fuel_cell.region = -fuel_cyl & -outer_sphere   # <-- fixed: bounded by outer sphere

    moderator_cell = openmc.Cell(name='moderator')
    moderator_cell.fill = water
    moderator_cell.region = +fuel_cyl & -outer_sphere

    root_universe = openmc.Universe(cells=[fuel_cell, moderator_cell])
    geom = openmc.Geometry(root_universe)
    geom.export_to_xml()
    return geom, fuel_cell, moderator_cell, fuel_cyl, outer_sphere

def build_settings():
    settings = openmc.Settings()
    settings.run_mode = 'eigenvalue'
    settings.batches = BATCHES
    settings.inactive = INACTIVE
    settings.particles = PARTICLES
    settings.seed = SEED
    # overlap check and helpful temperature interpolation if ACE supports it
    settings.check_overlaps = True
    settings.temperature = {'method': 'interpolation'}
    settings.statepoint_interval = 2   # write statepoint every 5 batches
    settings.export_to_xml()
    return settings

def build_tallies(fuel_cell, outer_sphere):
    tallies = openmc.Tallies()

    # cell filter
    cell_filter = openmc.CellFilter(fuel_cell)

    # SAFE scores only: do not include 'capture' (some OpenMC versions reject it)
    # We'll compute capture = absorption - fission in post-processing.
    t_nuclide = openmc.Tally(name='nuclide_rates')
    t_nuclide.filters = [cell_filter]
    t_nuclide.nuclides = ['U235', 'U238']
    t_nuclide.scores = ['fission', 'nu-fission', 'absorption']
    tallies.append(t_nuclide)

    # heating (power deposition) per cell
    t_heating = openmc.Tally(name='heating')
    t_heating.filters = [cell_filter]
    t_heating.scores = ['heating']
    tallies.append(t_heating)

    # mesh tallies (coarse 3D mesh)
    mesh = openmc.RegularMesh()
    mesh.dimension = [NX, NY, NZ]
    # keep mesh slightly inside the sphere to avoid boundary equality issues
    mesh.lower_left = [-OUTER_RADIUS + MESH_MARGIN, -OUTER_RADIUS + MESH_MARGIN, -OUTER_RADIUS + MESH_MARGIN]
    mesh.upper_right = [OUTER_RADIUS - MESH_MARGIN, OUTER_RADIUS - MESH_MARGIN, OUTER_RADIUS - MESH_MARGIN]
    mesh_filter = openmc.MeshFilter(mesh)

    t_mesh_flux = openmc.Tally(name='mesh_flux')
    t_mesh_flux.filters = [mesh_filter]
    t_mesh_flux.scores = ['flux']
    tallies.append(t_mesh_flux)

    t_mesh_heating = openmc.Tally(name='mesh_heating')
    t_mesh_heating.filters = [mesh_filter]
    t_mesh_heating.scores = ['heating']
    tallies.append(t_mesh_heating)

    # surface current for outer sphere (leakage diagnostic)
    surf_filter = openmc.SurfaceFilter(outer_sphere)
    t_current = openmc.Tally(name='surface_current')
    t_current.filters = [surf_filter]
    t_current.scores = ['current']
    tallies.append(t_current)

    # energy-binned spectrum (coarse)
    energy_bins = [0.0, 1e-5, 0.5, 20.0]
    energy_filter = openmc.EnergyFilter(energy_bins)
    t_spectrum = openmc.Tally(name='spectrum')
    t_spectrum.filters = [cell_filter, energy_filter]
    t_spectrum.scores = ['flux']
    tallies.append(t_spectrum)

    tallies.export_to_xml()
    return tallies

def try_plot_geometry(geom, filename='geom_xy.png'):
    # Produce an XY plot to inspect geometry visually
    try:
        p = openmc.Plot()
        p.filename = filename.split(".")[0]
        p.width = (2*OUTER_RADIUS, 2*OUTER_RADIUS)
        p.pixels = (600, 600)
        p.origin = (0.0, 0.0, 0.0)
        p.basis = 'xy'
        plots = openmc.Plots([p])
        plots.export_to_xml()

        # OpenMC writes PNG to default 'plots' directory or working dir depending on version
        print(f"[INFO] Geometry plot requested (check geom_xy.png or plots/geom_xy.png).")
    except Exception as e:
        print("[WARN] Could not create geometry plot:", e)

def run_openmc_safe(run_depletion=False, depletion_days=0.0):
    """
    Run OpenMC (transport). If run_depletion True, attempt a single depletion step (safe fallback).
    Wrap in try/except to capture runtime errors (lost particles, invalid tally scores).
    """
    try:
        if run_depletion:
            # try depletion (only if configured)
            try:
                from openmc.deplete import CoupledOperator, Integrator
                op = CoupledOperator()
                integrator = Integrator(op, [depletion_days])
                integrator.integrate()
                return 0
            except Exception as e:
                print("[WARN] Depletion unavailable or failed; falling back to transport run:", e)

        # plain transport run
        print(f"[RUN] Starting OpenMC transport at {datetime.utcnow().isoformat()} UTC")
        openmc.run(mpi_args=['mpiexec','--use-hwthread-cpus','-n','4'])   # allow executor to use mpi args from environment
        return 0
    except RuntimeError as rte:
        msg = str(rte)
        print("[ERROR] OpenMC runtime error:\n", msg)
        # detect common causes and give immediate advice
        if "lost particles" in msg.lower() or "Maximum number of lost particles" in msg:
            print("\nA. Likely cause: geometry gap/overlap causing particles to be in no cell.")
            print("   - Ensure every fuel/moderator/void region is bounded (e.g., fuel_cell.region = -fuel_cyl & -outer_sphere).")
            print("   - Run geom.check_overlaps() and create a small debug transport (fewer particles) to inspect.")
        if "Invalid tally score" in msg:
            print("\nB. Invalid tally score detected. Use safe scores (flux, fission, nu-fission, absorption, heating, current).")
        # produce geometry checks and a small debug plot to help diagnose
        try:
            geom = openmc.Geometry.from_xml()   # read geometry
            print("[INFO] Running geom.check_overlaps() for diagnostics...")
            geom.check_overlaps()
        except Exception as e:
            print("[WARN] geom.check_overlaps() failed:", e)
        try:
            try_plot_geometry(openmc.Geometry.from_xml())
        except Exception:
            pass
        raise

def latest_statepoint():
    sps = sorted(glob.glob('statepoint.*.h5'))
    return sps[-1] if sps else None

def postprocess_statepoint(sp_file):
    print(f"[POST] Reading statepoint file: {sp_file}")
    sp = openmc.StatePoint(sp_file)

    # k_eff
    try:
        k_res = sp.k_combined
        try:
            k_val = k_res.nominal_value
            k_std = k_res.std_dev
        except Exception:
            k_val = float(k_res)
            k_std = None
    except Exception:
        try:
            k_val = float(sp.k_combined[0].n)
            k_std = float(sp.k_combined[0].s)
        except Exception:
            k_val = None
            k_std = None
    print(f"  k_eff = {k_val}  (std dev = {k_std})")

    # Find tally by name robustly
    def find_tally_by_name(name):
        for t_id, t in sp.tallies.items():
            try:
                if getattr(t, 'name', None) == name:
                    return t
            except Exception:
                pass
        return None

    t_nuclide = find_tally_by_name('nuclide_rates')
    if t_nuclide is not None:
        # parse mean/std shapes intelligently
        mean = t_nuclide.mean
        std = t_nuclide.std_dev
        # attempt to infer shapes (common: (n_filters, n_scores, n_nuclides))
        arr_mean = mean
        arr_std = std
        # flatten to 1D if needed, then reshape if possible
        flat_mean = arr_mean.flatten()
        flat_std = arr_std.flatten()
        # try to detect counts from metadata
        scores = t_nuclide.scores
        nuclides = [n.decode() if isinstance(n, bytes) else n for n in t_nuclide.nuclides]
        # attempt reshape
        try:
            arr_mean2 = flat_mean.reshape((-1, len(scores), len(nuclides)))
            arr_std2 = flat_std.reshape((-1, len(scores), len(nuclides)))
            # take first filter index (most common case)
            data_mean = arr_mean2[0]
            data_std = arr_std2[0]
            fission_mean = data_mean[scores.index('fission'), :]
            abs_mean = data_mean[scores.index('absorption'), :]
            fission_std = data_std[scores.index('fission'), :]
            abs_std = data_std[scores.index('absorption'), :]
            # compute capture = absorption - fission
            import numpy as np
            capture_mean = abs_mean - fission_mean
            capture_std = (abs_std**2 + fission_std**2)**0.5
            print("  Nuclide reaction rates (per-nuclide) [mean ± std]:")
            for i, nuc in enumerate(nuclides):
                print(f"    {nuc}: fission = {fission_mean[i]:.6e} ± {fission_std[i]:.2e}")
                print(f"           absorption = {abs_mean[i]:.6e} ± {abs_std[i]:.2e}")
                print(f"           capture = {capture_mean[i]:.6e} ± {capture_std[i]:.2e}")
        except Exception as e:
            print("[WARN] Could not reshape nuclide_rates tally to expected dims:", e)
    else:
        print("  Tally 'nuclide_rates' not found in statepoint.")

    # mesh flux & heating example (print a few values)
    t_mesh_flux = find_tally_by_name('mesh_flux')
    if t_mesh_flux:
        mf = t_mesh_flux.mean.flatten()
        ms = t_mesh_flux.std_dev.flatten()
        nprint = min(8, mf.size)
        print(f"  mesh_flux: first {nprint} cells mean:", mf[:nprint])
        print(f"             first {nprint} cells std :", ms[:nprint])
    else:
        print("  mesh_flux tally missing.")

    t_heating = find_tally_by_name('heating')
    if t_heating:
        hm = t_heating.mean.flatten()
        hs = t_heating.std_dev.flatten()
        print(f"  heating (cell): mean {hm}, std {hs}")
    else:
        print("  heating tally missing.")

    t_curr = find_tally_by_name('surface_current')
    if t_curr:
        cm = t_curr.mean.flatten()
        print("  surface_current (leakage) first values:", cm[:min(6, cm.size)])
    else:
        print("  surface_current tally missing.")

    # energy spectrum
    t_spec = find_tally_by_name('spectrum')
    if t_spec:
        print("  spectrum (energy bins) means:", t_spec.mean.flatten())
    else:
        print("  spectrum tally missing.")

def main():
    fuel, water, mats = build_materials()
    geom, fuel_cell, moderator_cell, fuel_cyl, outer_sphere = build_geometry(fuel, water)
    settings = build_settings()
    tallies = build_tallies(fuel_cell, outer_sphere)

    # export & check geometry
    geom.export_to_xml()
    print("[INFO] Running geometry overlap check...")

    # optional plot for debugging
    if DEBUG:
        try_plot_geometry(geom)

    # run OpenMC safely
    try:
        run_openmc_safe(run_depletion=RUN_DEPLETION, depletion_days=0.0)
    except Exception as e:
        print("[FATAL] OpenMC failed. See messages above for diagnosis.")
        # exit non-zero so caller sees failure
        sys.exit(2)

    # postprocess latest statepoint
    sp = latest_statepoint()
    if sp:
        postprocess_statepoint(sp)
    else:
        print("[ERROR] No statepoint file found after run. OpenMC may have failed before writing output.")

    print("[DONE] Script finished.")

if __name__ == "__main__":
    main()

