#!/usr/bin/env python

# compute_power_from_statepoint.py
import glob, math, numpy as np
import openmc

# constant
E_FISSION_J = 200e6 * 1.602176634e-19  # J per fission (~3.204353268e-11 J)

def latest_statepoint():
    sps = sorted(glob.glob('statepoint.*.h5'))
    return sps[-1] if sps else None

def read_total_power_from_statepoint(sp_file, power_target_W=None):
    """
    Reads the statepoint file and returns (P_mean_W, P_std_W, notes).
    If a heating tally exists (mesh_heating or heating), use it; else use fission tally.
    If power_target_W provided, scale tallies to match that physical power (normalization).
    """
    sp = openmc.StatePoint(sp_file)

    # helper: find tally by name
    def find_tally(name):
        for t_id, t in sp.tallies.items():
            try:
                if getattr(t, "name", "") == name:
                    return t
            except Exception:
                pass
        return None

    # prefer heating tally
    heating = find_tally('heating') or find_tally('mesh_heating')
    if heating is not None:
        mean = heating.mean.flatten().sum()   # sum over cells -> per-source energy deposition
        std  = heating.std_dev.flatten()
        sigma = math.sqrt((std**2).sum()) if std.size>0 else None
        # mean is energy per source particle (or per history) -> need to convert to W
        # If we have a power_target_W use it as normalization scale
        if power_target_W is not None:
            # scale factor such that mean * scale = power_target
            scale = power_target_W / mean if mean != 0 else 0.0
            P = power_target_W
            P_std = sigma * scale if sigma is not None else None
            return P, P_std, "heating tally scaled to power_target"
        else:
            # if unknown normalization, return per-source energy (J per source)
            return mean, sigma, "heating (J per source); provide power_target to scale to W"
    # else fallback to fission tally
    f_tally = find_tally('nuclide_rates') or find_tally('fission_rate') or find_tally('fission')
    if f_tally is not None:
        # assume fission score is present; find index for 'fission' in scores
        scores = f_tally.scores
        try:
            idx = scores.index('fission')
        except ValueError:
            # sometimes 'nu-fission' is used
            try:
                idx = scores.index('nu-fission')
            except ValueError:
                idx = None
        if idx is None:
            return None, None, "No fission/nu-fission score found in fission tally"
        mean = f_tally.mean.flatten()
        std  = f_tally.std_dev.flatten()
        # We need to reshape based on scores/nuclides ordering; do a robust approach:
        flat_mean = mean.flatten()
        flat_std  = std.flatten()
        # If ordering unknown, try to locate a single-score case
        if flat_mean.size == 1:
            f_mean = float(flat_mean[0])
            f_std = float(flat_std[0])
        else:
            # If multiple: attempt to reshape (n_filters, n_scores, n_nuclides)
            try:
                n_scores = len(scores)
                # assume single filter and single nuclide dimension or multiple nuclides appended last
                f_mean = flat_mean[idx]
                f_std  = flat_std[idx]
            except Exception:
                f_mean = float(flat_mean[0])
                f_std  = float(flat_std[0])
        # now convert to power
        P_per_source = f_mean * E_FISSION_J
        P_std_per_source = f_std * E_FISSION_J
        if power_target_W is not None:
            scale = power_target_W / P_per_source if P_per_source != 0 else 0.0
            P = power_target_W
            P_std = P_std_per_source * scale if P_std_per_source is not None else None
            return P, P_std, "fission tally scaled to power_target"
        else:
            return P_per_source, P_std_per_source, "power in J-per-source (provide power_target to scale to W)"

    return None, None, "No suitable heating or fission tally found in statepoint"

if __name__ == "__main__":
    sp = latest_statepoint()
    if sp is None:
        print("No statepoint found.")
    else:
        # example: if your reactor rated thermal power is known, set power_target_W
        rated_MWth = 3000.0  # example
        P, Pstd, note = read_total_power_from_statepoint(sp, power_target_W=rated_MWth * 1e6)
        print("Result:", P, "W ±", Pstd, "| note:", note)

