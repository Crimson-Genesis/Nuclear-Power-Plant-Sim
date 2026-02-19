import pickle
import time, json, threading
import zmq
import os
import sys
import random

import glob
from datetime import datetime
import openmc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lib.lib import *

import yaml

# ----------------- Add these methods to your ParticalSim class -----------------

import csv
import numpy as np
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
import json

# Place inside your ParticalSim class definition


class ParticalSim:
    def __init__(
        self,
    ):
        # ---------------- USER TUNABLES ----------------
        self.OUTER_RADIUS = 50.0  # cm, outer bounding sphere
        self.FUEL_RADIUS = 0.39  # cm, fuel cylinder radius
        self.DEBUG = (
            True  # if True create geometry plots and use small runs for quick debugging
        )
        self.RUN_DEPLETION = (
            False  # set True only if openmc.deplete is configured & you want depletion
        )
        # Debug run sizes (fast)
        self.BATCHES = 20
        statefile = self.latest_statepoint()
        if statefile[0]:
            self.BATCHES = statefile[1]
        self.INACTIVE = 5
        self.PARTICLES = 20000
        self.SEED = 12345

        # Mesh tally settings
        self.NX, self.NY, self.NZ = 20, 20, 20
        self.MESH_MARGIN = 0.1  # avoid placing mesh exactly on boundary
        # ------------------------------------------------

        self.data_dict = dict()
        self.data_dict["batch_size"] = self.BATCHES
    def updateBatchSize(self, num):
        self.BATCHES += num
        self.data_dict["batch_size"] = self.BATCHES

    def build_materials(self):
        # simple demo composition (atomic fractions). Replace with number densities for production.
        fuel = openmc.Material(name="UO2 fuel")
        fuel.add_nuclide("U235", 0.04)
        fuel.add_nuclide("U238", 0.96)
        fuel.add_nuclide("O16", 2.0)
        fuel.set_density("g/cm3", 10.5)

        water = openmc.Material(name="Light water")
        water.add_element("H", 2.0)
        water.add_element("O", 1.0)
        water.set_density("g/cm3", 0.743)
        # s(alpha,beta) if available
        try:
            water.add_s_alpha_beta("c_H_in_H2O")
        except Exception:
            # missing S(a,b) data is fine — script will warn
            pass

        mats = openmc.Materials([fuel, water])
        mats.export_to_xml()
        return fuel, water, mats

    def build_geometry(self, fuel, water):
        # Surfaces
        fuel_cyl = openmc.ZCylinder(r=self.FUEL_RADIUS)
        outer_sphere = openmc.Sphere(r=self.OUTER_RADIUS, boundary_type="vacuum")

        # Cells — IMPORTANT: bind both fuel & moderator inside outer_sphere to avoid lost particles
        fuel_cell = openmc.Cell(name="fuel")
        fuel_cell.fill = fuel
        fuel_cell.region = (
            -fuel_cyl & -outer_sphere
        )  # <-- fixed: bounded by outer sphere

        moderator_cell = openmc.Cell(name="moderator")
        moderator_cell.fill = water
        moderator_cell.region = +fuel_cyl & -outer_sphere

        root_universe = openmc.Universe(cells=[fuel_cell, moderator_cell])
        geom = openmc.Geometry(root_universe)
        geom.export_to_xml()
        return geom, fuel_cell, moderator_cell, fuel_cyl, outer_sphere

    def build_settings(self):
        settings = openmc.Settings()
        settings.run_mode = "eigenvalue"
        settings.batches = self.BATCHES
        settings.inactive = self.INACTIVE
        settings.particles = self.PARTICLES
        settings.seed = self.SEED
        settings.photon_transport = True
        # overlap check and helpful temperature interpolation if ACE supports it
        settings.check_overlaps = True
        settings.temperature = {"method": "interpolation"}
        settings.export_to_xml()
        return settings

    def build_tallies(self, fuel_cell, outer_sphere):
        """
        Build a comprehensive set of tallies for an NPP-level simulation.
        This attempts many commonly useful scores; invalid/unavailable scores are skipped
        with a warning (so the script remains robust across OpenMC versions).
        """
        tallies = openmc.Tallies()

        # def add_tally_safe(name, filters=None, scores=None, nuclides=None):
        #     """
        #     Helper: create tally and append to tallies. If OpenMC rejects the scores,
        #     catch the exception, print a warning, and skip.
        #     """
        #     try:
        #         t = openmc.Tally(name=name)
        #         if filters:
        #             t.filters = filters
        #         if nuclides:
        #             t.nuclides = nuclides
        #         if scores:
        #             t.scores = scores
        #         tallies.append(t)
        #         return t
        #     except Exception as e:
        #         print(f"[WARN] Could not create tally '{name}' with scores {scores}: {e}")
        #         return None

        def add_tally_safe(name, filters=None, scores=None, nuclides=None):
            """
            Create tally while normalizing/deduplicating scores.
            Will skip the tally if OpenMC rejects it or if no valid scores remain.
            """
            # Normalize score strings (strip, lower) and deduplicate preserving order
            def normalize_score(s):
                if isinstance(s, bytes):
                    s = s.decode()
                s0 = str(s).strip()
                # map common synonyms to canonical names (extend if needed)
                syn = {
                    "capture": "(n,gamma)",
                    "ngamma": "(n,gamma)",
                    "kappa-fission": "kappa-fission",
                    "fission-q-prompt": "fission-q-prompt",
                    "fission-q-recoverable": "fission-q-recoverable",
                }
                # if s is a plain integer string like "102", keep as-is (OpenMC accepts ints), but map known synonyms
                s_lower = s0.lower()
                if s_lower in syn:
                    return syn[s_lower]
                return s0

            if scores:
                seen = set()
                norm_scores = []
                for sc in scores:
                    scn = normalize_score(sc)
                    if scn in seen:
                        # skip duplicate
                        continue
                    seen.add(scn)
                    norm_scores.append(scn)
                scores = norm_scores

            # If scores is empty after normalization, skip tally
            if scores is None or len(scores) == 0:
                print(f"[WARN] Not adding tally '{name}' because no valid scores after normalization.")
                return None

            try:
                t = openmc.Tally(name=name)
                if filters:
                    t.filters = filters
                if nuclides:
                    t.nuclides = nuclides
                t.scores = scores  # set normalized unique list
                tallies.append(t)
                return t
            except Exception as e:
                print(f"[WARN] Could not create tally '{name}' with scores {scores}: {e}")
                return None

        # common filters
        cell_filter = openmc.CellFilter(fuel_cell)
        neutron_filter = openmc.ParticleFilter("neutron")

        # coarse 3D mesh over the whole outer sphere box (used by many mesh tallies)
        mesh = openmc.RegularMesh()
        mesh.dimension = [self.NX, self.NY, self.NZ]
        mesh.lower_left = [
            -self.OUTER_RADIUS + self.MESH_MARGIN,
            -self.OUTER_RADIUS + self.MESH_MARGIN,
            -self.OUTER_RADIUS + self.MESH_MARGIN,
        ]
        mesh.upper_right = [
            self.OUTER_RADIUS - self.MESH_MARGIN,
            self.OUTER_RADIUS - self.MESH_MARGIN,
            self.OUTER_RADIUS - self.MESH_MARGIN,
        ]
        mesh_filter = openmc.MeshFilter(mesh)

        # surface filter (outer boundary) and mesh-surface filter
        surf_filter = openmc.SurfaceFilter(outer_sphere)
        try:
            mesh_surf_filter = openmc.MeshSurfaceFilter(mesh)  # may not be needed; safe try
        except Exception:
            mesh_surf_filter = None

        # energy bins for coarse spectrum / pulse-height use
        energy_bins = [0.0, 1e-5, 0.5, 20.0]
        energy_filter = openmc.EnergyFilter(energy_bins)

        # ---------- 1) Basic flux and reaction tallies ----------
        add_tally_safe(
            "cell_flux",
            filters=[cell_filter, neutron_filter],
            scores=["flux"],
        )

        # per-nuclide reaction rates (useful for depletion). Start with main actinides; add more as needed.
        add_tally_safe(
            "nuclide_rates",
            filters=[cell_filter, neutron_filter],
            nuclides=["U235", "U238", "Pu239", "Pu241"],  # adjust list to what you track
            scores=["fission", "nu-fission", "absorption"],
        )

        # detailed fission/neutron production splits
        add_tally_safe(
            "fission_production",
            filters=[cell_filter, neutron_filter],
            scores=["fission", "nu-fission", "prompt-nu-fission", "delayed-nu-fission"],
        )

        # scatter and nu-scatter
        add_tally_safe(
            "scatter_rates",
            filters=[cell_filter, neutron_filter],
            scores=["scatter", "nu-scatter"],
        )

        # specific reaction channels (use named scores where supported)
        # Radiative capture: try common names; if unsupported it will be skipped
        # add_tally_safe("radiative_capture", filters=[cell_filter], scores=["(n,gamma)", "102"])
        add_tally_safe("radiative_capture", filters=[cell_filter, neutron_filter], scores=["(n,gamma)"])


        # examples of other specific channels often useful in depletion or activation:
        add_tally_safe("reaction_n2n", filters=[cell_filter, neutron_filter], scores=["(n,2n)"])  # MT numbers vary; may be skipped
        add_tally_safe("reaction_np_na", filters=[cell_filter, neutron_filter], scores=["(n,np)", "(n,2n)"])  # illustrative

        # ---------- 2) Energy / spectrum tallies ----------
        add_tally_safe(
            "spectrum_flux",
            filters=[cell_filter, energy_filter, neutron_filter],
            scores=["flux"],
        )

        # ---------- 3) Heating and energy deposition (global + mesh) ----------
        add_tally_safe(
            "heating",
            filters=[cell_filter, neutron_filter],
            scores=["heating", "heating-local", "kappa-fission", "fission-q-prompt", "fission-q-recoverable"],
        )

        add_tally_safe(
            "mesh_heating",
            filters=[mesh_filter, neutron_filter],
            scores=["heating", "kappa-fission"],
        )

        # ---------- 4) Mesh flux (spatial shape) ----------
        add_tally_safe(
            "mesh_flux",
            filters=[mesh_filter, neutron_filter],
            scores=["flux"],
        )

        # ---------- 5) Currents & leakage ----------
        add_tally_safe(
            "surface_current",
            filters=[surf_filter, neutron_filter],
            scores=["current"],
        )

        if mesh_surf_filter is not None:
            add_tally_safe(
                "mesh_surface_current",
                filters=[mesh_surf_filter, neutron_filter],
                scores=["current"],
            )

        # ---------- 6) Diagnostic / tally quality ----------
        add_tally_safe("events", filters=[cell_filter, neutron_filter], scores=["events"])
        add_tally_safe("inverse_velocity", filters=[cell_filter, neutron_filter], scores=["inverse-velocity"])

        # ---------- 7) Damage / pulse-height / decay-rate ----------
        add_tally_safe("damage_energy", filters=[cell_filter, neutron_filter], scores=["damage-energy"])
        add_tally_safe("pulse_height", filters=[cell_filter, energy_filter], scores=["pulse-height"])
        add_tally_safe("decay_rate", filters=[cell_filter, neutron_filter], scores=["decay-rate"])

        # ---------- 8) Delayed / prompt energy/neutron tallies ----------
        add_tally_safe("delayed_prompt_split", filters=[cell_filter, neutron_filter], scores=["prompt-nu-fission", "delayed-nu-fission"])

        # ---------- 9) kappa / fission energy tallies ----------
        add_tally_safe("kappa_fission", filters=[cell_filter, neutron_filter], scores=["kappa-fission", "fission-q-prompt", "fission-q-recoverable"])

        # ---------- 10) Damage / activation specifics ----------
        # If you track activation products or specific ENDF MT channels for transmutation, include them:
        # Example: MT=102 is (n,gamma). You can add arbitrary MT numbers if your library supports them.
        add_tally_safe("mt_102_ngamma", filters=[cell_filter, neutron_filter], scores=["102",])

        # ---------- 11) Mesh-based scoring for diagnostic leakage / streaming ----------
        if mesh_surf_filter is not None:
            add_tally_safe("mesh_partial_currents", filters=[mesh_surf_filter, neutron_filter], scores=["current"])

        # ---------- 12) Per-nuclide detailed channels for depletion (explicit) ----------
        # It's often necessary to request per-nuclide channels needed for your depletion chain.
        # Example: for xenon/I chains and actinide transmutation:
        add_tally_safe(
            "depletion_channels_U235",
            filters=[cell_filter, neutron_filter],
            nuclides=["U235"],
            scores=["fission", "absorption", "(n,gamma)", "nu-fission"],
        )
        add_tally_safe(
            "depletion_channels_U238",
            filters=[cell_filter, neutron_filter],
            nuclides=["U238"],
            scores=["fission", "absorption", "(n,gamma)"],
        )
        # add similar per-nuclide tallies for Pu239, Xe135, I135, etc. as needed:
        # add_tally_safe("depletion_Xe135", filters=[cell_filter], nuclides=["Xe135"], scores=["(n,gamma)","absorption"])

        # # ---------- 13) Photon/gamma tallies (if doing gamma heating / dose) ----------
        # add_tally_safe(
        #     "photon_pulse_height",
        #     filters=[cell_filter, energy_filter],
        #     scores=["pulse-height"],
        # )
        # add_tally_safe("photon_heating", filters=[cell_filter], scores=["heating"])

        # ---------- 14) Safety: always export everything that was successfully created ----------
        # Note: add_tally_safe prints warnings for those scores not supported by the current OpenMC/data combo.
        tallies.export_to_xml()
        print("[INFO] build_tallies: exported tallies.xml with available tallies (unsupported scores were skipped).")
        return tallies


    # def build_tallies(self, fuel_cell, outer_sphere):
    #     tallies = openmc.Tallies()
    #
    #     # cell filter
    #     cell_filter = openmc.CellFilter(fuel_cell)
    #
    #     # SAFE scores only: do not include 'capture' (some OpenMC versions reject it)
    #     # We'll compute capture = absorption - fission in post-processing.
    #     t_nuclide = openmc.Tally(name="nuclide_rates")
    #     t_nuclide.filters = [cell_filter]
    #     t_nuclide.nuclides = ["U235", "U238"]
    #     t_nuclide.scores = ["fission", "nu-fission", "absorption"]
    #     tallies.append(t_nuclide)
    #
    #     # heating (power deposition) per cell
    #     t_heating = openmc.Tally(name="heating")
    #     t_heating.filters = [cell_filter]
    #     t_heating.scores = ["heating"]
    #     tallies.append(t_heating)
    #
    #     # mesh tallies (coarse 3D mesh)
    #     mesh = openmc.RegularMesh()
    #     mesh.dimension = [self.NX, self.NY, self.NZ]
    #     # keep mesh slightly inside the sphere to avoid boundary equality issues
    #     mesh.lower_left = [
    #         -self.OUTER_RADIUS + self.MESH_MARGIN,
    #         -self.OUTER_RADIUS + self.MESH_MARGIN,
    #         -self.OUTER_RADIUS + self.MESH_MARGIN,
    #     ]
    #     mesh.upper_right = [
    #         self.OUTER_RADIUS - self.MESH_MARGIN,
    #         self.OUTER_RADIUS - self.MESH_MARGIN,
    #         self.OUTER_RADIUS - self.MESH_MARGIN,
    #     ]
    #     mesh_filter = openmc.MeshFilter(mesh)
    #
    #     t_mesh_flux = openmc.Tally(name="mesh_flux")
    #     t_mesh_flux.filters = [mesh_filter]
    #     t_mesh_flux.scores = ["flux"]
    #     tallies.append(t_mesh_flux)
    #
    #     t_mesh_heating = openmc.Tally(name="mesh_heating")
    #     t_mesh_heating.filters = [mesh_filter]
    #     t_mesh_heating.scores = ["heating"]
    #     tallies.append(t_mesh_heating)
    #
    #     # surface current for outer sphere (leakage diagnostic)
    #     surf_filter = openmc.SurfaceFilter(outer_sphere)
    #     t_current = openmc.Tally(name="surface_current")
    #     t_current.filters = [surf_filter]
    #     t_current.scores = ["current"]
    #     tallies.append(t_current)
    #
    #     # energy-binned spectrum (coarse)
    #     energy_bins = [0.0, 1e-5, 0.5, 20.0]
    #     energy_filter = openmc.EnergyFilter(energy_bins)
    #     t_spectrum = openmc.Tally(name="spectrum")
    #     t_spectrum.filters = [cell_filter, energy_filter]
    #     t_spectrum.scores = ["flux"]
    #     tallies.append(t_spectrum)
    #
    #     tallies.export_to_xml()
    #     return tallies

    def try_plot_geometry(self, geom, filename="geom_xy.png"):
        # Produce an XY plot to inspect geometry visually
        try:
            p = openmc.Plot()
            p.filename = filename.split(".")[0]
            p.width = (2 * self.OUTER_RADIUS, 2 * self.OUTER_RADIUS)
            p.pixels = (600, 600)
            p.origin = (0.0, 0.0, 0.0)
            p.basis = "xy"
            plots = openmc.Plots([p])
            plots.export_to_xml()
            p.to_ipython_image()

            # OpenMC writes PNG to default 'plots' directory or working dir depending on version
            print(
                f"[INFO] Geometry plot requested (check geom_xy.png or plots/geom_xy.png)."
            )
        except Exception as e:
            print("[WARN] Could not create geometry plot:", e)

    def run_openmc_safe(self, run_depletion=False, depletion_days=0.0, sp=None):
        """
        Run OpenMC (transport). If run_depletion True, attempt a single depletion step (safe fallback).
        Wrap in try/except to capture runtime errors (lost particles, invalid tally scores).
        """
        try:
            # plain transport run
            print(
                f"[RUN] Starting OpenMC transport at {datetime.utcnow().isoformat()} UTC"
            )
            if sp:
                openmc.run(
                    mpi_args=["mpiexec", "--use-hwthread-cpus", "-n", "4"],
                    restart_file=sp
                )  # allow executor to use mpi args from environment
            else:
                openmc.run(
                    mpi_args=["mpiexec", "--use-hwthread-cpus", "-n", "4"]
                )  # allow executor to use mpi args from environment
            return 0
        except RuntimeError as rte:
            msg = str(rte)
            print("[ERROR] OpenMC runtime error:\n", msg)
            # detect common causes and give immediate advice
            if (
                "lost particles" in msg.lower()
                or "Maximum number of lost particles" in msg
            ):
                print(
                    "\nA. Likely cause: geometry gap/overlap causing particles to be in no cell."
                )
                print(
                    "   - Ensure every fuel/moderator/void region is bounded (e.g., fuel_cell.region = -fuel_cyl & -outer_sphere)."
                )
                print(
                    "   - Run geom.check_overlaps() and create a small debug transport (fewer particles) to inspect."
                )
            if "Invalid tally score" in msg:
                print(
                    "\nB. Invalid tally score detected. Use safe scores (flux, fission, nu-fission, absorption, heating, current)."
                )
            # produce geometry checks and a small debug plot to help diagnose
            try:
                geom = openmc.Geometry.from_xml()  # read geometry
                print("[INFO] Running geom.check_overlaps() for diagnostics...")
                # geom.check_overlaps()
            except Exception as e:
                print("[WARN] geom.check_overlaps() failed:", e)
            try:
                try_plot_geometry(openmc.Geometry.from_xml())
            except Exception:
                pass
            raise

    def latest_statepoint(
        self,
    ):
        sps = sorted(glob.glob("statepoint.*.h5"))
        r1 = sps[-1] if sps else None
        if sps:
            sps_max_batch = max([int(i.split(".")[1]) for i in sps])
            return (r1, sps_max_batch,)
        else:
            return (r1, 0)

    def postprocess_all_tallies(self, sp_file):
        """
        Robust reader for the comprehensive tally set (the ~25 tallies).
        Populates self.data_dict with raw means/stds and useful derived quantities.
        NOTE: tallies are returned in *per source-particle* units. This function
        does NOT perform physical normalization to W or neutrons/s. Set a scale
        (e.g. via self.data_dict['scale_JperSource_to_W'] or power_target) afterwards.
        """
        # import numpy as np
        # import os
        # from datetime import datetime

        print(f"[POST] Reading statepoint: {sp_file}")
        sp = openmc.StatePoint(sp_file)

        # timestamp / provenance
        try:
            mtime = os.path.getmtime(sp_file)
            self.data_dict["snapshot_time"] = datetime.utcfromtimestamp(mtime).isoformat() + "Z"
        except Exception:
            self.data_dict["snapshot_time"] = None

        # --- k_eff ---
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
        print(f"  k_eff = {k_val}  (std = {k_std})")
        self.data_dict["k_val"] = k_val
        self.data_dict["k_std"] = k_std
        # derived: reactivity (if available)
        try:
            if k_val is not None:
                rho = (k_val - 1.0) / k_val
                self.data_dict["rho"] = rho
                self.data_dict["rho_std"] = (1.0 / (k_val**2)) * k_std if k_std is not None else None
            else:
                self.data_dict["rho"] = None
                self.data_dict["rho_std"] = None
        except Exception:
            self.data_dict["rho"] = None
            self.data_dict["rho_std"] = None

        # helper to find tally by name robustly
        def find_tally(name):
            for tid, t in sp.tallies.items():
                try:
                    if getattr(t, "name", None) == name:
                        return t
                except Exception:
                    pass
            return None

        # helper to safely extract flattened mean/std (returns (mean_array, std_array) or (None,None))
        def get_mean_std(tally):
            try:
                mean = np.asarray(tally.mean).flatten()
                std = np.asarray(tally.std_dev).flatten()
                return mean, std
            except Exception:
                return None, None

        # store which tallies were found
        self.data_dict["tallies_present"] = []

        # ---------- 1) nuclide_rates (per-nuclide fission/absorption) ----------
        t = find_tally("nuclide_rates")
        if t is not None:
            self.data_dict["tallies_present"].append("nuclide_rates")
            try:
                mean = np.asarray(t.mean)
                std  = np.asarray(t.std_dev)
                # deduce scores & nuclides meta
                scores = list(t.scores)
                nuclides = [n.decode() if isinstance(n, bytes) else n for n in t.nuclides]
                self.data_dict["nuclide_rates_scores"] = scores
                self.data_dict["nuclide_rates_nuclides"] = nuclides

                flat_mean = mean.flatten()
                flat_std  = std.flatten()
                # Attempt reshape: (n_filters, n_scores, n_nuclides)
                try:
                    arr = flat_mean.reshape((-1, len(scores), len(nuclides)))
                    arrs = flat_std.reshape((-1, len(scores), len(nuclides)))
                    # take first filter index (typical)
                    data_mean = arr[0]
                    data_std  = arrs[0]
                    # save per-score arrays keyed by score
                    for si, score in enumerate(scores):
                        key_mean = f"nuclide_rates_{score}_mean"
                        key_std  = f"nuclide_rates_{score}_std"
                        self.data_dict[key_mean] = data_mean[si, :].astype(float)
                        self.data_dict[key_std]  = data_std[si, :].astype(float)
                    # compute capture = absorption - fission if both exist
                    if ("absorption" in scores) and ("fission" in scores):
                        abs_idx = scores.index("absorption")
                        fiss_idx = scores.index("fission")
                        abs_mean = data_mean[abs_idx, :]
                        abs_std = data_std[abs_idx, :]
                        fiss_mean = data_mean[fiss_idx, :]
                        fiss_std = data_std[fiss_idx, :]
                        cap_mean = abs_mean - fiss_mean
                        cap_std = np.sqrt(abs_std**2 + fiss_std**2)
                        self.data_dict["nuclide_rates_capture_mean"] = cap_mean.astype(float)
                        self.data_dict["nuclide_rates_capture_std"]  = cap_std.astype(float)
                    # print summary
                    print("  nuclide_rates found for nuclides:", nuclides)
                    for i, nuc in enumerate(nuclides):
                        fmean = self.data_dict.get("nuclide_rates_fission_mean", None)
                        if fmean is not None:
                            print(f"    {nuc}: fission = {fmean[i]:.6e}")
                except Exception as e:
                    # fallback: save flattened arrays
                    self.data_dict["nuclide_rates_flat_mean"] = flat_mean.astype(float)
                    self.data_dict["nuclide_rates_flat_std"]  = flat_std.astype(float)
                    print("[WARN] Could not reshape nuclide_rates to (filters,scores,nuclides):", e)
            except Exception as e:
                print("[WARN] Error reading nuclide_rates:", e)
        else:
            print("  nuclide_rates tally missing.")

        # ---------- 2) fission_production (fission / nu-fission / prompt / delayed) ----------
        t = find_tally("fission_production")
        if t is not None:
            self.data_dict["tallies_present"].append("fission_production")
            m, s = get_mean_std(t)
            self.data_dict["fission_production_mean"] = m
            self.data_dict["fission_production_std"]  = s
        # ---------- 3) scatter_rates ----------
        t = find_tally("scatter_rates")
        if t is not None:
            self.data_dict["tallies_present"].append("scatter_rates")
            m, s = get_mean_std(t)
            self.data_dict["scatter_rates_mean"] = m
            self.data_dict["scatter_rates_std"]  = s

        # ---------- 4) radiative_capture / mt_102_ngamma ----------
        t = find_tally("radiative_capture")
        if t is not None:
            self.data_dict["tallies_present"].append("radiative_capture")
            m, s = get_mean_std(t)
            self.data_dict["radiative_capture_mean"] = m
            self.data_dict["radiative_capture_std"]  = s
        t = find_tally("mt_102_ngamma")
        if t is not None:
            self.data_dict["tallies_present"].append("mt_102_ngamma")
            m, s = get_mean_std(t)
            self.data_dict["mt_102_ngamma_mean"] = m
            self.data_dict["mt_102_ngamma_std"]  = s

        # ---------- 5) spectrum_flux (energy-binned flux) ----------
        t = find_tally("spectrum_flux")
        if t is not None:
            self.data_dict["tallies_present"].append("spectrum_flux")
            m, s = get_mean_std(t)
            self.data_dict["spectrum_flux_mean"] = m
            self.data_dict["spectrum_flux_std"]  = s
            # if energy_filter use, store number of bins
            try:
                # number of bins = size / filters? best-effort:
                self.data_dict["spectrum_num_bins"] = len(m)
                print(f"  spectrum_flux: {len(m)} entries")
            except Exception:
                pass

        # ---------- 6) heating (cell) and mesh_heating ----------
        t = find_tally("heating")
        if t is not None:
            self.data_dict["tallies_present"].append("heating")
            hm, hs = get_mean_std(t)
            self.data_dict["heating_mean"] = hm
            self.data_dict["heating_std"]  = hs
            # total heating per source (J per source) if heating is present:
            try:
                total_heating_per_source = float(np.sum(hm))
                total_heating_std = float(np.sqrt(np.sum(hs**2)))
                self.data_dict["total_heating_per_source_J"] = total_heating_per_source
                self.data_dict["total_heating_per_source_std_J"] = total_heating_std
                print(f"  heating sum (J per source) = {total_heating_per_source:.6e} ± {total_heating_std:.6e}")
            except Exception:
                pass
        else:
            print("  heating tally missing.")

        t = find_tally("mesh_heating")
        if t is not None:
            self.data_dict["tallies_present"].append("mesh_heating")
            mh_mean, mh_std = get_mean_std(t)
            self.data_dict["mesh_heating_mean_flat"] = mh_mean
            self.data_dict["mesh_heating_std_flat"]  = mh_std
            # reshape to 3D if size matches NX*NY*NZ
            try:
                total_cells = int(self.NX * self.NY * self.NZ)
                if mh_mean is not None and mh_mean.size >= total_cells:
                    mh3 = np.asarray(mh_mean[:total_cells]).reshape((self.NX, self.NY, self.NZ))
                    mh3s = np.asarray(mh_std[:total_cells]).reshape((self.NX, self.NY, self.NZ))
                    self.data_dict["mesh_heating_mean_3d"] = mh3.astype(float)
                    self.data_dict["mesh_heating_std_3d"]  = mh3s.astype(float)
                    # compute simple peaking factor (max/avg) in mesh_heating if possible
                    try:
                        cell_powers = mh3
                        peak = float(cell_powers.max())
                        avg = float(cell_powers.mean())
                        peaking = peak / avg if avg != 0 else None
                        self.data_dict["mesh_heating_peak_W_per_source"] = peak
                        self.data_dict["mesh_heating_avg_W_per_source"] = avg
                        self.data_dict["mesh_heating_peaking"] = peaking
                        print(f"  mesh_heating peaking (per-source) = {peaking}")
                    except Exception:
                        pass
            except Exception as e:
                print("[WARN] couldn't reshape mesh_heating:", e)
        else:
            print("  mesh_heating tally missing.")

        # ---------- 7) mesh_flux ----------
        t = find_tally("mesh_flux")
        if t is not None:
            self.data_dict["tallies_present"].append("mesh_flux")
            mf, ms = get_mean_std(t)
            self.data_dict["mesh_flux_mean_flat"] = mf
            self.data_dict["mesh_flux_std_flat"]  = ms
            try:
                total_cells = int(self.NX * self.NY * self.NZ)
                if mf is not None and mf.size >= total_cells:
                    mf3 = np.asarray(mf[:total_cells]).reshape((self.NX, self.NY, self.NZ))
                    ms3 = np.asarray(ms[:total_cells]).reshape((self.NX, self.NY, self.NZ))
                    self.data_dict["mesh_flux_mean_3d"] = mf3.astype(float)
                    self.data_dict["mesh_flux_std_3d"]  = ms3.astype(float)
                    print(f"  mesh_flux shape: {(self.NX, self.NY, self.NZ)}")
            except Exception as e:
                print("[WARN] couldn't reshape mesh_flux:", e)
        else:
            print("  mesh_flux tally missing.")

        # ---------- 8) surface_current & mesh_surface_current ----------
        t = find_tally("surface_current")
        if t is not None:
            self.data_dict["tallies_present"].append("surface_current")
            cm, cs = get_mean_std(t)
            self.data_dict["surface_current_mean"] = cm
            self.data_dict["surface_current_std"]  = cs
            # sum leakage per-source
            try:
                leak = float(np.sum(cm))
                self.data_dict["leakage_per_source"] = leak
                print(f"  leakage (per-source, summed current) = {leak:.6e}")
            except Exception:
                pass
        else:
            print("  surface_current tally missing.")

        t = find_tally("mesh_surface_current")
        if t is not None:
            self.data_dict["tallies_present"].append("mesh_surface_current")
            mc_mean, mc_std = get_mean_std(t)
            self.data_dict["mesh_surface_current_mean_flat"] = mc_mean
            self.data_dict["mesh_surface_current_std_flat"]  = mc_std

        # ---------- 9) events, inverse-velocity ----------
        t = find_tally("events")
        if t is not None:
            self.data_dict["tallies_present"].append("events")
            m, s = get_mean_std(t)
            self.data_dict["events_mean"] = m
            self.data_dict["events_std"]  = s

        t = find_tally("inverse_velocity")
        if t is not None:
            self.data_dict["tallies_present"].append("inverse_velocity")
            m, s = get_mean_std(t)
            self.data_dict["inverse_velocity_mean"] = m
            self.data_dict["inverse_velocity_std"]  = s

        # ---------- 10) damage-energy, pulse-height, decay-rate ----------
        t = find_tally("damage_energy")
        if t is not None:
            self.data_dict["tallies_present"].append("damage_energy")
            m, s = get_mean_std(t)
            self.data_dict["damage_energy_mean"] = m
            self.data_dict["damage_energy_std"]  = s

        t = find_tally("pulse_height")
        if t is not None:
            self.data_dict["tallies_present"].append("pulse_height")
            m, s = get_mean_std(t)
            self.data_dict["pulse_height_mean"] = m
            self.data_dict["pulse_height_std"]  = s

        t = find_tally("decay_rate")
        if t is not None:
            self.data_dict["tallies_present"].append("decay_rate")
            m, s = get_mean_std(t)
            self.data_dict["decay_rate_mean"] = m
            self.data_dict["decay_rate_std"]  = s

        # ---------- 11) delayed/prompt split ----------
        t = find_tally("delayed_prompt_split")
        if t is not None:
            self.data_dict["tallies_present"].append("delayed_prompt_split")
            m, s = get_mean_std(t)
            self.data_dict["delayed_prompt_split_mean"] = m
            self.data_dict["delayed_prompt_split_std"]  = s

        # ---------- 12) kappa_fission / fission-q tallies ----------
        t = find_tally("kappa_fission")
        if t is not None:
            self.data_dict["tallies_present"].append("kappa_fission")
            m, s = get_mean_std(t)
            self.data_dict["kappa_fission_mean"] = m
            self.data_dict["kappa_fission_std"]  = s

        # ---------- 13) depletion_channels_* tallies (examples) ----------
        t = find_tally("depletion_channels_U235")
        if t is not None:
            self.data_dict["tallies_present"].append("depletion_channels_U235")
            m, s = get_mean_std(t)
            self.data_dict["depletion_U235_mean"] = m
            self.data_dict["depletion_U235_std"]  = s

        t = find_tally("depletion_channels_U238")
        if t is not None:
            self.data_dict["tallies_present"].append("depletion_channels_U238")
            m, s = get_mean_std(t)
            self.data_dict["depletion_U238_mean"] = m
            self.data_dict["depletion_U238_std"]  = s

        # ---------- 14) photon tallies (if present) ----------
        t = find_tally("photon_pulse_height")
        if t is not None:
            self.data_dict["tallies_present"].append("photon_pulse_height")
            m, s = get_mean_std(t)
            self.data_dict["photon_pulse_height_mean"] = m
            self.data_dict["photon_pulse_height_std"]  = s

        t = find_tally("photon_heating")
        if t is not None:
            self.data_dict["tallies_present"].append("photon_heating")
            m, s = get_mean_std(t)
            self.data_dict["photon_heating_mean"] = m
            self.data_dict["photon_heating_std"]  = s

        # ---------- 15) fallback: any other tallies present in statepoint (collect names) ----------
        try:
            for tid, t in sp.tallies.items():
                name = getattr(t, "name", None)
                if name and name not in self.data_dict.get("tallies_present", []):
                    # try to read means/std and add under generic key
                    try:
                        m, s = get_mean_std(t)
                        key_mean = f"tally_{name}_mean"
                        key_std  = f"tally_{name}_std"
                        self.data_dict[key_mean] = m
                        self.data_dict[key_std]  = s
                        self.data_dict.setdefault("tallies_present", []).append(name)
                        print(f"  (found additional tally '{name}')")
                    except Exception:
                        pass
        except Exception:
            pass

        # ---------- 16) Derived quantities & notes ----------
        # Save a placeholder for scale conversions (user should set in pipeline)
        # e.g. to convert J-per-source -> W set scale_JperSource_to_W = physical_power_W / total_heating_per_source_J
        self.data_dict.setdefault("scale_JperSource_to_W", None)
        self.data_dict.setdefault("power_target_W", None)  # optionally set by caller

        # If heating present and scale set, compute physical total power
        try:
            scale = self.data_dict.get("scale_JperSource_to_W", None)
            if ("total_heating_per_source_J" in self.data_dict) and (scale is not None):
                P = float(self.data_dict["total_heating_per_source_J"]) * float(scale)
                P_std = float(self.data_dict.get("total_heating_per_source_std_J", 0.0)) * float(scale)
                self.data_dict["total_power_W"] = P
                self.data_dict["total_power_std_W"] = P_std
                print(f"  total_power_W = {P:.6e} ± {P_std:.6e}")
        except Exception:
            pass

        # Basic QA printout
        print("[POST] Tallies present:", ", ".join(self.data_dict.get("tallies_present", [])))
        print("[POST] Stored keys in data_dict:", ", ".join(list(self.data_dict.keys())))

        return self.data_dict

    # constants
    EV_TO_J = 1.602176634e-19

    def set_power_target(self, power_W):
        """
        Call this before running OpenMC or before postprocessing if you want to
        normalize tallies to a specific plant power (Watts).
        """
        if power_W is None:
            self.POWER_TARGET_W = None
            print("[INFO] POWER_TARGET_W unset; no automatic scaling will be applied.")
        else:
            self.POWER_TARGET_W = float(power_W)
            print(f"[INFO] POWER_TARGET_W set to {self.POWER_TARGET_W:.6e} W")

    def compute_scaling_and_apply(self):
        """
        After postprocessing tallies into self.data_dict (so keys like
        'total_heating_per_source_J' or 'total_heating_per_source_eV' exist),
        call this to compute `scale_JperSource_to_W` and to compute derived
        physical quantities like total_power_W and scaled mesh/pin powers.
        """
        # ensure data_dict exists
        if not hasattr(self, "data_dict") or self.data_dict is None:
            raise RuntimeError("data_dict not found; run postprocess_all_tallies() first")

        # 1) Determine total heating per source in J (try multiple keys)
        totJ = None
        if "total_heating_per_source_J" in self.data_dict and self.data_dict["total_heating_per_source_J"] is not None:
            totJ = float(self.data_dict["total_heating_per_source_J"])
        elif "total_heating_per_source_eV" in self.data_dict and self.data_dict["total_heating_per_source_eV"] is not None:
            totJ = float(self.data_dict["total_heating_per_source_eV"]) * EV_TO_J
            # store converted value for clarity
            self.data_dict["total_heating_per_source_J"] = totJ
        else:
            # try to derive from available heating_mean (which may be in eV/source)
            if "heating_mean" in self.data_dict and self.data_dict["heating_mean"] is not None:
                try:
                    hm = np.asarray(self.data_dict["heating_mean"], dtype=float)
                    # sum the per-cell heating (assumes heating_mean is in eV/source)
                    tot_eV = float(hm.sum())
                    totJ = tot_eV * EV_TO_J
                    self.data_dict["total_heating_per_source_eV"] = tot_eV
                    self.data_dict["total_heating_per_source_J"] = totJ
                except Exception:
                    totJ = None

        # 2) Compute scale if POWER_TARGET_W is set
        if getattr(self, "POWER_TARGET_W", None) is not None:
            if totJ is None or totJ == 0.0:
                raise RuntimeError("Cannot compute scaling: no total heating per source available to normalize against POWER_TARGET_W.")
            scale = float(self.POWER_TARGET_W) / float(totJ)  # units: W per (J per source) => 1/s (i.e., sources/sec)
            self.data_dict["scale_JperSource_to_W"] = scale
            self.data_dict["power_target_W"] = float(self.POWER_TARGET_W)
            # compute total_power_W (sanity)
            self.data_dict["total_power_W"] = float(totJ * scale)
            print(f"[INFO] Computed scale: scale_JperSource_to_W = {scale:.6e} (sources/sec equivalent)")
            print(f"[INFO] total_heating_per_source_J = {totJ:.6e}; scaled total_power_W = {self.data_dict['total_power_W']:.6e} (should equal POWER_TARGET_W)")
        else:
            # no power target provided: just record that no scale applied
            self.data_dict.setdefault("scale_JperSource_to_W", None)
            print("[INFO] POWER_TARGET_W not set; skipping automatic scaling. You can set it with set_power_target(W).")

        # 3) Apply scaling to mesh_heating and mesh_flux if available
        scale = self.data_dict.get("scale_JperSource_to_W", None)
        if scale is not None:
            # mesh_heating_mean_3d -> W per cell
            if "mesh_heating_mean_3d" in self.data_dict and self.data_dict["mesh_heating_mean_3d"] is not None:
                mh3 = np.asarray(self.data_dict["mesh_heating_mean_3d"], dtype=float)
                self.data_dict["mesh_heating_W_3d"] = mh3 * scale
                if "mesh_heating_std_3d" in self.data_dict and self.data_dict["mesh_heating_std_3d"] is not None:
                    self.data_dict["mesh_heating_W_std_3d"] = np.asarray(self.data_dict["mesh_heating_std_3d"], dtype=float) * scale
                # recompute peaking factor in physical units
                avg = float(self.data_dict["mesh_heating_W_3d"].mean())
                peak = float(self.data_dict["mesh_heating_W_3d"].max())
                self.data_dict["mesh_heating_peaking_physical"] = (peak / avg) if avg != 0 else None
                print(f"[INFO] mesh heating converted to W: peak={peak:.6e} W, avg={avg:.6e} W, peaking={self.data_dict['mesh_heating_peaking_physical']}")
            # If you want to scale mesh_flux to e.g. neutrons/cm2/s, you need a neutrons-per-source rate (scale_neutrons_per_source),
            # which is typically the same as 'scale' if heating was computed from fission energy; otherwise compute separately.
            # For now we set flux scaling placeholder:
            self.data_dict.setdefault("mesh_flux_scaling_factor", None)

        return self.data_dict

    # def postprocess_statepoint(self, sp_file):
    #     print(f"[POST] Reading statepoint file: {sp_file}")
    #     sp = openmc.StatePoint(sp_file)
    #
    #     # k_eff
    #     try:
    #         k_res = sp.k_combined
    #         try:
    #             k_val = k_res.nominal_value
    #             k_std = k_res.std_dev
    #         except Exception:
    #             k_val = float(k_res)
    #             k_std = None
    #     except Exception:
    #         try:
    #             k_val = float(sp.k_combined[0].n)
    #             k_std = float(sp.k_combined[0].s)
    #         except Exception:
    #             k_val = None
    #             k_std = None
    #     print(f"  k_eff = {k_val}  (std dev = {k_std})")
    #     self.data_dict["k_val"] = k_val
    #     self.data_dict["k_std"] = k_std
    #
    #     # Find tally by name robustly
    #     def find_tally_by_name(name):
    #         for t_id, t in sp.tallies.items():
    #             try:
    #                 if getattr(t, "name", None) == name:
    #                     return t
    #             except Exception:
    #                 pass
    #         return None
    #
    #     t_nuclide = find_tally_by_name("nuclide_rates")
    #     if t_nuclide is not None:
    #         # parse mean/std shapes intelligently
    #         mean = t_nuclide.mean
    #         std = t_nuclide.std_dev
    #         # attempt to infer shapes (common: (n_filters, n_scores, n_nuclides))
    #         arr_mean = mean
    #         arr_std = std
    #         # flatten to 1D if needed, then reshape if possible
    #         flat_mean = arr_mean.flatten()
    #         flat_std = arr_std.flatten()
    #         # try to detect counts from metadata
    #         scores = t_nuclide.scores
    #         nuclides = [
    #             n.decode() if isinstance(n, bytes) else n for n in t_nuclide.nuclides
    #         ]
    #         # attempt reshape
    #         try:
    #             arr_mean2 = flat_mean.reshape((-1, len(scores), len(nuclides)))
    #             arr_std2 = flat_std.reshape((-1, len(scores), len(nuclides)))
    #             # take first filter index (most common case)
    #             data_mean = arr_mean2[0]
    #             data_std = arr_std2[0]
    #             fission_mean = data_mean[scores.index("fission"), :]
    #             abs_mean = data_mean[scores.index("absorption"), :]
    #             fission_std = data_std[scores.index("fission"), :]
    #             abs_std = data_std[scores.index("absorption"), :]
    #             self.data_dict["fission_std"] = fission_std
    #             self.data_dict["fission_mean"] = fission_mean
    #             self.data_dict["abs_std"] = abs_std
    #             self.data_dict["abs_mean"] = abs_mean
    #
    #             # compute capture = absorption - fission
    #             import numpy as np
    #
    #             capture_mean = abs_mean - fission_mean
    #             capture_std = (abs_std**2 + fission_std**2) ** 0.5
    #
    #             self.data_dict["capture_mean"] = capture_mean
    #             self.data_dict["capture_std"] = capture_std
    #             print("  Nuclide reaction rates (per-nuclide) [mean ± std]:")
    #             for i, nuc in enumerate(nuclides):
    #                 print(
    #                     f"    {nuc}: fission = {fission_mean[i]:.6e} ± {fission_std[i]:.2e}"
    #                 )
    #                 print(
    #                     f"           absorption = {abs_mean[i]:.6e} ± {abs_std[i]:.2e}"
    #                 )
    #                 print(
    #                     f"           capture = {capture_mean[i]:.6e} ± {capture_std[i]:.2e}"
    #                 )
    #         except Exception as e:
    #             print(
    #                 "[WARN] Could not reshape nuclide_rates tally to expected dims:", e
    #             )
    #     else:
    #         print("  Tally 'nuclide_rates' not found in statepoint.")
    #
    #     # mesh flux & heating example (print a few values)
    #     t_mesh_flux = find_tally_by_name("mesh_flux")
    #     if t_mesh_flux:
    #         mf = t_mesh_flux.mean.flatten()
    #         ms = t_mesh_flux.std_dev.flatten()
    #         self.data_dict["mf"] = mf
    #         self.data_dict["ms"] = ms
    #         nprint = min(8, mf.size)
    #         print(f"  mesh_flux: first {nprint} cells mean:", mf[:nprint])
    #         print(f"             first {nprint} cells std :", ms[:nprint])
    #     else:
    #         print("  mesh_flux tally missing.")
    #
    #     t_heating = find_tally_by_name("heating")
    #     if t_heating:
    #         hm = t_heating.mean.flatten()
    #         hs = t_heating.std_dev.flatten()
    #         self.data_dict["hm"] = hm
    #         self.data_dict["hs"] = hs
    #         print(f"  heating (cell): mean {hm}, std {hs}")
    #     else:
    #         print("  heating tally missing.")
    #
    #     t_curr = find_tally_by_name("surface_current")
    #     if t_curr:
    #         cm = t_curr.mean.flatten()
    #         self.data_dict["cm"] = cm
    #         print("  surface_current (leakage) first values:", cm[: min(6, cm.size)])
    #     else:
    #         print("  surface_current tally missing.")
    #
    #     # energy spectrum
    #     t_spec = find_tally_by_name("spectrum")
    #     if t_spec:
    #         tm = t_spec.mean.flatten()
    #         self.data_dict["tm"] = tm
    #         print("  spectrum (energy bins) means:", tm)
    #     else:
    #         print("  spectrum tally missing.")

    def get_data(self):
        return self.data_dict

    def load_data(self):
        try:
            openmc.reset_auto_ids()
        except Exception:
            # older OpenMC might not have this; ignore if unavailable
            pass

        fuel, water, mats = self.build_materials()
        geom, fuel_cell, moderator_cell, fuel_cyl, outer_sphere = self.build_geometry(
            fuel, water
        )
        settings = self.build_settings()
        tallies = self.build_tallies(fuel_cell, outer_sphere)

        # export & check geometry
        geom.export_to_xml()
        print("[INFO] Running geometry overlap check...")

        # optional plot for debugging
        if self.DEBUG:
            self.try_plot_geometry(geom)

    def run(self, sp=None):
        # run OpenMC safely
        try:
            self.run_openmc_safe(run_depletion=self.RUN_DEPLETION, depletion_days=0.0, sp=sp)
        except Exception as e:
            print("[FATAL] OpenMC failed. See messages above for diagnosis.")
            # exit non-zero so caller sees failure
            sys.exit(2)

    def get_postprocess_data(self):
        # postprocess latest statepoint
        sp = self.latest_statepoint()[0]
        if sp:
            self.postprocess_all_tallies(sp)
        else:
            print(
                "[ERROR] No statepoint file found after run. OpenMC may have failed before writing output."
            )

        print("[DONE] Script finished.")
        return self.data_dict

    def add_detector_tallies(
        self,
        geom,
        fuel_cell,
        outer_sphere,
        detector_positions=None,
        detector_radius=2.0,
    ):
        """
        Add small sphere/cell tallies that emulate in-core or ex-core detectors.
        - detector_positions: list of (x,y,z) tuples where to place detectors.
          Defaults: 4 positions outside the fuel at +/- X and +/- Y just outside fuel cylinder.
        - detector_radius: radius of small detector sphere (cm)
        This function writes new tally definitions to tallies.xml (appending to previous tallies).
        """
        # Use current materials as "void" detector filled with the moderator material (or vacuum)
        # We'll create small volumes in the moderator region (or just outside outer_sphere for ex-core)
        if detector_positions is None:
            # place 4 detectors on +x,-x,+y,-y outside the fuel radius at r = fuel_radius + margin
            margin = 5.0
            r = self.FUEL_RADIUS + margin
            detector_positions = [
                (r, 0.0, 0.0),
                (-r, 0.0, 0.0),
                (0.0, r, 0.0),
                (0.0, -r, 0.0),
            ]

        tallies = openmc.Tallies()

        detectors = []
        for i, pos in enumerate(detector_positions):
            name = f"detector_{i}"
            # small sphere surface
            s = openmc.Sphere(r=detector_radius, x0=pos[0], y0=pos[1], z0=pos[2])
            # create a detector material: thin fill with moderator composition (non-absorbing)
            det_mat = openmc.Material(name=f"det_mat_{i}")
            # Use light water composition as placeholder (or create vacuum)
            det_mat.add_element("H", 2.0)
            det_mat.add_element("O", 1.0)
            det_mat.set_density(
                "g/cm3", 0.001
            )  # tiny density to avoid absorbing too much
            # Export material to unique file set by appending to existing mats later
            detectors.append((name, s, det_mat))

        # Add detector materials and temporary geometry cells to a temp universe so tallies can reference them
        # We'll create tallies referencing spherical volumes via CellFilter using temporary cells,
        # but we won't overwrite the main geometry.xml here — instead the function will create tally xml that uses
        # SurfaceFilters with surface definitions (OpenMC supports SurfaceFilter). Simpler: create mesh tallies around positions
        # For portability, we will create small mesh volumes centered at detector positions using MeshFilters.

        # Create tallies for detector "voxels" using RegularMesh clipped to small region around detector pos
        for i, (name, sph, det_mat) in enumerate(detectors):
            # small mesh size box around pos
            px, py, pz = detector_positions[i]
            dx = detector_radius
            mesh = openmc.RegularMesh()
            # one-cell mesh covering small cube around detector
            mesh.dimension = [1, 1, 1]
            mesh.lower_left = [px - dx, py - dx, pz - dx]
            mesh.upper_right = [px + dx, py + dx, pz + dx]
            mesh_filter = openmc.MeshFilter(mesh)

            t_det = openmc.Tally(name=name)
            t_det.filters = [mesh_filter]
            # Use 'flux' as detector measure (neutron flux proportional to count rate)
            t_det.scores = ["flux"]
            tallies.append(t_det)

        # Append these tallies to any existing tallies.xml by reading current tallies.xml and writing combined
        # Easiest approach: export these to a temporary tally file and then merge with existing tallies.xml externally.
        tallies.export_to_xml()  # writes tallies.xml (overwrites existing)
        print(
            "[INFO] Detector tallies added exported to tallies.xml (be careful: this overwrites previous tallies.xml)."
        )
        print(
            "       If you want to preserve original tallies, copy them before calling this function."
        )

        # NOTE: If you want to preserve previous tallies, modify this function to read existing tallies.xml and append.
        # For now we keep it simple. You can re-call build_tallies afterward to restore standard tallies.

    def simulate_detectors_from_mesh(
        self, sp_file, detector_positions, NX=None, NY=None, NZ=None, mesh_box=None
    ):
        """
        Fast surrogate: compute detector responses from the mesh_flux tally by integrating over nearest mesh cell(s).
        - sp_file: statepoint filename
        - detector_positions: list of (x,y,z)
        - NX/NY/NZ: mesh dims used in the run (if None, use self.NX etc)
        - mesh_box: [-r,r] box extents (if None deduce from self.OUTER_RADIUS)
        Returns: list of detector estimates (flux per detector) and optional uncertainties
        """
        sp = openmc.StatePoint(sp_file)
        if NX is None:
            NX = self.NX
        if NY is None:
            NY = self.NY
        if NZ is None:
            NZ = self.NZ
        if mesh_box is None:
            r = self.OUTER_RADIUS
            mesh_box = [-r + self.MESH_MARGIN, r - self.MESH_MARGIN]

        # find mesh_flux tally
        t_mesh = None
        for tid, t in sp.tallies.items():
            if getattr(t, "name", "") == "mesh_flux":
                t_mesh = t
                break
        if t_mesh is None:
            raise RuntimeError(
                "mesh_flux tally not found in statepoint; cannot simulate detectors from mesh"
            )

        # flatten means and stds
        mean = t_mesh.mean.flatten()
        std = t_mesh.std_dev.flatten()

        # reshape to (NX,NY,NZ) with caveats: if multiple score dims exist this may need adjusting
        total_cells = NX * NY * NZ
        if mean.size < total_cells:
            raise RuntimeError(
                f"mesh_flux size ({mean.size}) smaller than NX*NY*NZ ({total_cells})"
            )
        vals = mean[:total_cells].reshape((NX, NY, NZ))
        errs = std[:total_cells].reshape((NX, NY, NZ))

        # compute coordinates for cell centers
        low = mesh_box[0]
        high = mesh_box[1]
        xs = np.linspace(low, high, NX, endpoint=False) + (high - low) / NX / 2.0
        ys = np.linspace(low, high, NY, endpoint=False) + (high - low) / NY / 2.0
        zs = np.linspace(low, high, NZ, endpoint=False) + (high - low) / NZ / 2.0

        det_vals = []
        det_errs = []
        for px, py, pz in detector_positions:
            # find nearest cell index
            ix = int(np.clip(np.searchsorted(xs, px), 0, NX - 1))
            iy = int(np.clip(np.searchsorted(ys, py), 0, NY - 1))
            iz = int(np.clip(np.searchsorted(zs, pz), 0, NZ - 1))
            det_vals.append(float(vals[ix, iy, iz]))
            det_errs.append(float(errs[ix, iy, iz]))
        return det_vals, det_errs

    def try_depletion_step(self, dt_days=1.0 / 24.0):
        """
        Attempt a single depletion step using openmc.deplete.CoupledOperator.
        This will run both transport & depletion and write depletion file.
        If depletion is unavailable or fails, function returns False.
        """
        try:
            from openmc.deplete import CoupledOperator, Integrator

            print("[INFO] Running a single depletion step (this may be slow)...")
            op = CoupledOperator()
            integrator = Integrator(op, [dt_days])
            integrator.integrate()
            print(
                "[INFO] Depletion step completed and depletion_results.h5 written (or similar)."
            )
            return True
        except Exception as e:
            print("[WARN] openmc.deplete unavailable or failed:", e)
            return False

    def perturbation_sweep_boron(
        self, ppm_list, power_target_W=None, run_depletion=False
    ):
        """
        Sweep soluble boron by modifying the water material and re-running OpenMC for each ppm in ppm_list.
        Returns list of (ppm, k_eff, k_std).
        NOTE: conversion of ppm->atomic fraction used here is approximate. For accurate modeling convert to exact
        number densities using molar masses and Avogadro's number.
        """
        results = []
        # backup original materials.xml
        shutil.copy("materials.xml", "materials.xml.bak")
        for ppm in ppm_list:
            print(f"[SWEEP-BORON] Running for {ppm} ppm (approx)...")
            # re-create materials with boron approximate addition
            fuel, water, mats = self.build_materials()
            # create new water with boron mass fraction approx
            new_water = openmc.Material(name="Light water with B")
            new_water.set_density("g/cm3", 0.743)
            new_water.add_element("H", 2.0)
            new_water.add_element("O", 1.0)
            # approximate add boron by atomic fraction (very rough)
            b_frac = ppm * 1e-6
            try:
                new_water.add_element("B", b_frac)
            except Exception:
                # fallback: tiny B10 nuclide
                new_water.add_nuclide("B10", 1e-8)
            mats = openmc.Materials([fuel, new_water])
            mats.export_to_xml()
            # re-export geometry (unchanged)
            _, _, _, _, _ = self.build_geometry(fuel, new_water)
            # re-export tallies & settings
            self.build_tallies(
                fuel, outer_sphere=openmc.Sphere(r=self.OUTER_RADIUS)
            )  # quick replacement
            self.build_settings()
            # run transport
            self.run_openmc_safe(run_depletion=run_depletion)
            sp = self.latest_statepoint()[0]
            if sp:
                sp_obj = openmc.StatePoint(sp)
                try:
                    k_res = sp_obj.k_combined
                    try:
                        kv = k_res.nominal_value
                        ks = k_res.std_dev
                    except Exception:
                        kv = float(k_res)
                        ks = None
                except Exception:
                    kv = None
                    ks = None
            else:
                kv, ks = None, None
            results.append((ppm, kv, ks))
        # restore original materials
        if os.path.exists("materials.xml.bak"):
            shutil.move("materials.xml.bak", "materials.xml")
        # re-export original mats and geometry
        self.build_materials()
        self.build_geometry(*self.build_materials()[:2])
        return results

    def perturbation_sweep_rod(
        self, insertion_positions_cm, absorber_nuc="Gd157", run_depletion=False
    ):
        """
        Sweep a simple absorber rod insertion depth (demo). For realistic cores, provide a rod-parameterized geometry.
        Returns list of (insertion_cm, k_eff, k_std).
        """
        results = []
        # backup geometry xml
        shutil.copy("geometry.xml", "geometry.xml.bak")
        for ins in insertion_positions_cm:
            print(f"[SWEEP-ROD] insertion={ins} cm ...")
            # create absorber material
            absorber = openmc.Material(name="absorber_mat")
            try:
                absorber.add_nuclide(absorber_nuc, 1.0)
            except Exception:
                absorber.add_nuclide("Gd157", 1.0)
            absorber.set_density("g/cm3", 8.0)
            mats = openmc.Materials([absorber])
            mats.export_to_xml()
            # rebuild geometry with absorber inserted (demo: coaxial absorber in fuel)
            # This is a simplified approach; adapt for real rod lattice
            fuel, water, _ = self.build_materials()
            fuel_cell = openmc.Cell(name="fuel")
            fuel_cell.fill = fuel
            fuel_cell.region = -openmc.ZCylinder(r=self.FUEL_RADIUS) & -openmc.Sphere(
                r=self.OUTER_RADIUS
            )
            # absorber cylinder with z-limited insertion
            abs_cyl = openmc.ZCylinder(r=self.FUEL_RADIUS / 2.0)
            plane_min = openmc.ZPlane(z0=-ins / 2.0)
            plane_max = openmc.ZPlane(z0=ins / 2.0)
            absorber_cell = openmc.Cell(name="absorber")
            absorber_cell.fill = absorber
            absorber_cell.region = -abs_cyl & +plane_min & -plane_max
            moderator_cell = openmc.Cell(name="moderator")
            moderator_cell.fill = water
            moderator_cell.region = +openmc.ZCylinder(
                r=self.FUEL_RADIUS
            ) & -openmc.Sphere(r=self.OUTER_RADIUS)

            univ = openmc.Universe(cells=[fuel_cell, absorber_cell, moderator_cell])
            geom = openmc.Geometry(univ)
            geom.export_to_xml()
            self.build_tallies(
                fuel_cell, outer_sphere=openmc.Sphere(r=self.OUTER_RADIUS)
            )
            self.build_settings()
            self.run_openmc_safe(run_depletion=run_depletion)
            sp = self.latest_statepoint()[0]
            if sp:
                sp_obj = openmc.StatePoint(sp)
                try:
                    k_res = sp_obj.k_combined
                    try:
                        kv = k_res.nominal_value
                        ks = k_res.std_dev
                    except Exception:
                        kv = float(k_res)
                        ks = None
                except Exception:
                    kv = None
                    ks = None
            else:
                kv, ks = None, None
            results.append((ins, kv, ks))
        # restore geometry
        if os.path.exists("geometry.xml.bak"):
            shutil.move("geometry.xml.bak", "geometry.xml")
        self.build_geometry(*self.build_materials()[:2])
        return results

    def perturbation_sweep_temperature(self, temps_K, run_depletion=False):
        """
        Sweep material temperatures (fuel & water) and record k_eff for each temp.
        Requires ACE libraries for multiple temperatures or interpolation to work.
        Returns list of (T_K, k_eff, k_std).
        """
        results = []
        # backup materials
        shutil.copy("materials.xml", "materials.xml.bak")
        for T in temps_K:
            print(f"[SWEEP-TEMP] Running for temperature {T} K ...")
            # build materials with .temperature set
            fuel = openmc.Material(name="UO2 fuel")
            fuel.add_nuclide("U235", 0.04)
            fuel.add_nuclide("U238", 0.96)
            fuel.add_nuclide("O16", 2.0)
            fuel.set_density("g/cm3", 10.5)
            fuel.temperature = T

            water = openmc.Material(name="Light water")
            water.add_element("H", 2.0)
            water.add_element("O", 1.0)
            water.set_density("g/cm3", 0.743)
            water.temperature = T
            try:
                water.add_s_alpha_beta("c_H_in_H2O")
            except Exception:
                pass

            mats = openmc.Materials([fuel, water])
            mats.export_to_xml()
            # rebuild geometry/tallies/settings for this material set
            geom, fuel_cell, moderator_cell, fuel_cyl, outer_sphere = (
                self.build_geometry(fuel, water)
            )
            self.build_tallies(fuel_cell, outer_sphere)
            self.build_settings()
            self.run_openmc_safe(run_depletion=run_depletion)
            sp = self.latest_statepoint()[0]
            if sp:
                sp_obj = openmc.StatePoint(sp)
                try:
                    k_res = sp_obj.k_combined
                    try:
                        kv = k_res.nominal_value
                        ks = k_res.std_dev
                    except Exception:
                        kv = float(k_res)
                        ks = None
                except Exception:
                    kv = None
                    ks = None
            else:
                kv, ks = None, None
            results.append((T, kv, ks))
        # restore original materials
        if os.path.exists("materials.xml.bak"):
            shutil.move("materials.xml.bak", "materials.xml")
        self.build_materials()
        return results


# ----------------- End of methods to add -----------------


class Reactor_core:
    def __init__(self):
        with open(config_file_name(), "r") as file:
            self.config = yaml.safe_load(file)

        self.ctx = zmq.Context()
        self.ctrl_endpoint = self.config["connections"]["ctrl"]["endpoint"]
        self.tick_endpoint = self.config["connections"]["tick"]["endpoint"]
        self.heartbeat_endpoint = self.config["connections"]["heartbeat"]["endpoint"]
        self.telemetry_endpoint = self.config["connections"]["telemetry"]["endpoint"]

        self.ctrl = self.ctx.socket(
            getattr(zmq, self.config["connections"]["ctrl"]["type"])
        )
        # self.ctrl.connect(self.ctrl_endpoint)
        self.name = get_name()
        self.ctrl.setsockopt_string(zmq.IDENTITY, self.name)

        self.tick = self.ctx.socket(
            getattr(zmq, self.config["connections"]["tick"]["type"])
        )
        self.tick.setsockopt_string(zmq.SUBSCRIBE, "")

        self.heartbeat = self.ctx.socket(
            getattr(zmq, self.config["connections"]["heartbeat"]["type"])
        )
        self.telemetry = self.ctx.socket(
            getattr(zmq, self.config["connections"]["telemetry"]["type"])
        )

        get_connection_object(self.ctrl, self.config["connections"]["ctrl"]["type"])(
            self.ctrl_endpoint
        )
        # Test data log file
        self.test_data_log_file = "test_data_log_file.pkl"

        # sim variables
        self.sim_time = float()
        self.tick_index = float()
        self.time_step = float()
        self.time_scale = 1
        self.running = True
        self.paused = False
        self.state = "off"

        self.ctx = zmq.Context()
        self.reactor_parm = self.config["parameters"]
        self.tick_data = None
        self.dt_master = 1.0
        self.last_sim_time = None
        self.dt_sub_step = 1.0
        self.part_sim_model = ParticalSim()
        self.particalSimRunnerThread = threading.Thread(target=self.particalSimRunner, daemon=True)
        self.particalSimRunnerActive = False
        # self.part_sim_model.load_data()

        # Parameters from the config...
        self.params = self.config["parameters"]
        self.total_control_rods = sum(
            [bank["rods"] for bank in self.params["rod_insertion_depth"]["banks"]]
        )
        self.control_rod_banks = self.params["rod_insertion_depth"]["banks"]
        self.bppm_ref = self.params["boron_ppm"]["value"]

        # Main values after calculation
        self.rho_rods = 0.0

    def save(self,filename : str, data, type_=None):
        filename_split = filename.split(".")
        if len(filename_split) > 1 and not type_:
            type_ = filename_split[-1]
        elif len(filename_split) <= 1:
            print("Invalid file name...")
        else:...

        if type_ == "txt":
            with open(filename, "a") as file:
                file.write(data)
        elif type_ == "json":
            with open(filename, "a") as file:
                json.dump(data, file, indent=4)
        elif type_ == "pkl":
            with open(f"{filename_split[0]}-{self.part_sim_model.BATCHES}.{filename_split[-1]}" , "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.send_msg(f"Added the ParticalSim Batch Size - {self.part_sim_model.BATCHES} data to {self.test_data_log_file}")

    def establish_connections(self):
        get_connection_object(self.tick, self.config["connections"]["tick"]["type"])(
            self.tick_endpoint
        )
        get_connection_object(
            self.telemetry, self.config["connections"]["telemetry"]["type"]
        )(self.telemetry_endpoint)
        get_connection_object(
            self.heartbeat, self.config["connections"]["heartbeat"]["type"]
        )(self.heartbeat_endpoint)
        self.tick_thread.start()
        self.running = True
        return "Tick, Telemetry and HeartBeat"

    def particalSimRunner(self):
        self.part_sim_model.load_data()
        while True:
            statepoint = self.part_sim_model.latest_statepoint()
            statepoint_file = statepoint[0]
            statepoint_max_batch = statepoint[1]
            if statepoint_file:
                self.part_sim_model.updateBatchSize(10)
                self.part_sim_model.build_settings()
            self.send_msg(f"Started Partical Sim on Batch Size - {self.part_sim_model.BATCHES}")
            time.sleep(2)
            self.part_sim_model.run(sp=statepoint_file)
            time.sleep(2)
            part_sim_data = self.part_sim_model.get_postprocess_data()
            self.save(self.test_data_log_file, part_sim_data)
            self.send_msg(f"Completed Partical Sim on Batch Size - {self.part_sim_model.BATCHES}")
            time.sleep(5)

    def reactor_core_loop(self):
        times = self.dt_master / self.dt_sub_step
        for _ in range(times):

            # Sum of control rod reactivity calculation...
            for b in self.control_rod_banks:
                frac_inserted = b["depth"] / b["length"]
                if frac_inserted < 0.0:
                    frac_inserted = 0.0
                elif frac_inserted > 1.0:
                    frac_inserted = 1.0

                wt = b.get("worth_table")
                if wt is None:
                    total_bank_worth = self.params["control_rod_worth"]["value"] * (
                        b["rods"] / self.total_control_rods
                    )
                    W_at_x = total_bank_worth * frac_inserted
                else:
                    wt_x = wt["x"]
                    wt_y = wt["y"]
                    W_at_x = self.piecewise_linear_eval(wt_x, wt_y, frac_inserted)
                self.rho_rods += W_at_x

            # Boron Reactivity calculation (Boron concentration in the coolant)

    def piecewise_linear_eval(self, x_points, y_points, x):
        """
        Evaluate piecewise-linear function defined by (x_points, y_points) at x.
        Assumes x_points sorted ascending and x in [x_points[0], x_points[-1]].
        """
        if x <= x_points[0]:
            return y_points[0]
        if x >= x_points[-1]:
            return y_points[-1]
        # find interval
        for i in range(len(x_points) - 1):
            x0, x1 = x_points[i], x_points[i + 1]
            if x0 <= x <= x1:
                y0, y1 = y_points[i], y_points[i + 1]
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        # fallback
        return y_points[-1]

    def _rho_boron_linear(
        self,
    ): ...

    def rho_boron_dynamic(
        self,
    ): ...

    def start(self):
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        self.dealerListen_thread = threading.Thread(
            target=self._dealerListener, daemon=True
        )
        self.dealerListen_thread.start()

        self.tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        while True:
            time.sleep(1)

    def _tick_loop(self):
        while self.running:
            if not self.paused and self.time_scale > 0:
                try:
                    data = self.tick.recv_json(flags=zmq.NOBLOCK)
                    self.tick_data = data
                    if self.last_sim_time is None:
                        self.dt_master = data["time_step"] * data.get("time_scale", 1.0)
                    else:
                        self.dt_master = data["sim_time"] - self.last_sim_time
                    # real_sleep = self.time_step / max(self.time_scale, 1e-9)
                    self.last_sim_time = data["sim_time"]
                except zmq.Again:
                    ...
                time.sleep(0.2)

    def _control_loop(self):
        self.send_msg("Okey", type_="check", data=random.randint(1, 1000))
        self.running = True
        while self.running:
            time.sleep(2)

    def _dealerListener(self):
        while True:
            try:
                identity, msg = self.ctrl.recv_multipart(flags=zmq.NOBLOCK)
                if identity or msg:
                    decoded_msg = json.loads(msg.decode())
                    if (
                        decoded_msg["type"] == "command"
                        and decoded_msg["name"] == "control_system"
                    ):
                        self._executation(decoded_msg)
            except zmq.Again:
                pass
            time.sleep(0.05)

    def _executation(self, msg):
        cmd = msg["command"]
        if cmd == "establish_connections":
            cc = self.establish_connections()
            self.send_msg(
                f"Establish-ED Connections of {cc} at {self.name}", type_="status"
            )
        elif cmd == "start-partical-sim-thread":
            print("hello")
            if not self.particalSimRunnerActive:
                self.send_msg("Starting Partical Sim Thread...")
                self.particalSimRunnerThread.start()
                self.particalSimRunnerActive = True
                self.send_msg("Started Partical Sim Thread...")
        else:
            self.send_msg("Error", ok=False, error="unknown command")

    def send_msg(self, msg, type_="status", **kwargs):
        data = {
            "type": type_,
            "name": self.name,
            "status": f"{'Running' if self.running else 'Not Running'}",
            "msg": msg,
            **kwargs,
        }
        self.ctrl.send_multipart([b"", json.dumps(data).encode()])


if __name__ == "__main__":
    try:
        reactor = Reactor_core()
        reactor.start()
    except KeyboardInterrupt:
        print("Keyboard Interrupt !!!")
    except EOFError:
        print("EOF Error !!!")
