"""Aligned-spin parameter estimation with bilby + dynesty (pre-registration stage 5, section 3).

Usage: python scripts/run_pe_bilby.py GW150914|GW151226 [--npool 4]
Checkpoints in results/pe_bilby/<event>/ (git-ignored). Writes results/pe_bilby_<event>.json and
results/pe_bilby_<event>_samples.npz (git-ignored).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import const, cosmo, psd, waveforms  # noqa: E402
import evcommon  # noqa: E402

FS = const.FS
FS_PE, DUR, PAD = 2048, 8, 2
F_MIN, F_MAX = 20.0, 896.0
SEED = 20261116
MC_RANGE = {"GW150914": (25.0, 40.0), "GW151226": (9.0, 10.5)}


def q90(x):
    a, b, c = np.percentile(x, [5, 50, 95])
    return {"median": float(b), "lo": float(a), "hi": float(c)}


def main(name, npool, dry=False):
    import bilby
    from bilby.core.prior import Cosine, PowerLaw, Sine, Uniform
    from bilby.gw.prior import AlignedSpin, UniformInComponentsChirpMass, UniformInComponentsMassRatio

    ev = evcommon.EVENTS[name]
    best = evcommon.best(name)
    t_trig = best["seg_start_s"] + best["i_h"] / FS          # seconds from file start (H1 peak)
    gps_trig = ev["file"] + t_trig
    i_a = int(round((t_trig - 6.0) * FS))
    gps_start = ev["file"] + i_a / FS
    ifos = bilby.gw.detector.InterferometerList([])
    for det in ("H1", "L1"):
        x = evcommon.load_strain(name, det)
        st = psd.valid_stretches(x)
        a, b = [s for s in st if s[0] <= i_a and i_a + DUR * FS <= s[1]][0]
        w = min(const.PSD_SPAN * FS, b - a)
        p0 = max(a, min(int(t_trig * FS) - w // 2, b - w))
        fw, pw = psd.welch_median(x[p0:p0 + w], FS)
        seg = resample_poly(x[i_a - PAD * FS:i_a + (DUR + PAD) * FS], 1, FS // FS_PE)[PAD * FS_PE:(PAD + DUR) * FS_PE]
        del x
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.minimum_frequency, ifo.maximum_frequency = F_MIN, F_MAX
        ifo.strain_data.roll_off = 0.4
        ifo.strain_data.set_from_time_domain_strain(seg, sampling_frequency=FS_PE, duration=DUR, start_time=gps_start)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=fw, psd_array=pw)
        ifos.append(ifo)

    pri = bilby.core.prior.PriorDict()
    pri["chirp_mass"] = UniformInComponentsChirpMass(*MC_RANGE[name], name="chirp_mass")
    pri["mass_ratio"] = UniformInComponentsMassRatio(0.125, 1.0, name="mass_ratio")
    pri["chi_1"] = AlignedSpin(a_prior=Uniform(0, 0.99), z_prior=Uniform(-1, 1), name="chi_1")
    pri["chi_2"] = AlignedSpin(a_prior=Uniform(0, 0.99), z_prior=Uniform(-1, 1), name="chi_2")
    pri["luminosity_distance"] = PowerLaw(alpha=2, minimum=50, maximum=2000, name="luminosity_distance")
    pri["dec"] = Cosine(name="dec")
    pri["ra"] = Uniform(0, 2 * np.pi, name="ra", boundary="periodic")
    pri["theta_jn"] = Sine(name="theta_jn")
    pri["psi"] = Uniform(0, np.pi, name="psi", boundary="periodic")
    pri["phase"] = Uniform(0, 2 * np.pi, name="phase", boundary="periodic")
    pri["geocent_time"] = Uniform(gps_trig - 0.1, gps_trig + 0.1, name="geocent_time")

    wg = bilby.gw.WaveformGenerator(
        duration=DUR, sampling_frequency=FS_PE, start_time=gps_start,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(waveform_approximant="IMRPhenomD", reference_frequency=20.0, minimum_frequency=F_MIN))
    like = bilby.gw.GravitationalWaveTransient(ifos, wg, priors=pri, distance_marginalization=True,
                                               phase_marginalization=True, time_marginalization=True)
    if dry:  # construction check: likelihood at the search best fit, maximised numerically over nothing
        like.parameters.update(dict(chirp_mass=waveforms.chirp_mass(best["m1"], best["m2"]), mass_ratio=best["m2"] / best["m1"],
                                    chi_1=best["chi"], chi_2=best["chi"], luminosity_distance=400.0, dec=-1.2, ra=1.4,
                                    theta_jn=2.8, psi=0.5, phase=0.0, geocent_time=gps_start, time_jitter=0.0))  # bilby marginalises time from the segment start
        rng = np.random.default_rng(0)
        bestl = -np.inf
        for _ in range(3000):  # crude random search over the sky and orientation
            like.parameters.update(dict(ra=rng.uniform(0, 2 * np.pi), dec=np.arcsin(rng.uniform(-1, 1)),
                                        psi=rng.uniform(0, np.pi), theta_jn=np.arccos(rng.uniform(-1, 1))))
            bestl = max(bestl, like.log_likelihood_ratio())
        print("max log likelihood ratio over 3000 random sky/orientation draws at the search best-fit masses:", bestl,
              " rho_net^2/2 =", best["rho_net"] ** 2 / 2)
        for ifo in ifos:
            print(ifo.name, "window factor", ifo.strain_data.window_factor, "n freq", int(ifo.frequency_mask.sum()))
        return
    outdir = ROOT / "results" / "pe_bilby" / name
    t0 = time.time()
    result = bilby.run_sampler(
        like, pri, sampler="dynesty", nlive=500, sample="acceptance-walk", naccept=30, dlogz=0.1,
        outdir=str(outdir), label=name, resume=True, check_point_delta_t=600, check_point_plot=False,
        npool=npool, seed=SEED, save="json", conversion_function=None)
    hours = (time.time() - t0) / 3600
    post = bilby.gw.conversion.generate_posterior_samples_from_marginalized_likelihood(result.posterior.copy(), like)
    mc, q = post["chirp_mass"].values, post["mass_ratio"].values
    m1, m2 = waveforms.masses_from_mc_q(mc, q)
    c1, c2 = post["chi_1"].values, post["chi_2"].values
    chieff = waveforms.chi_eff(m1, m2, c1, c2)
    dl = post["luminosity_distance"].values
    z = cosmo.z_from_dl(dl)
    np.savez(ROOT / "results" / f"pe_bilby_{name}_samples.npz", mc=mc, q=q, m1=m1, m2=m2, chi1=c1, chi2=c2,
             chieff=chieff, dl=dl, z=z, theta_jn=post["theta_jn"].values, ra=post["ra"].values, dec=post["dec"].values,
             geocent_time=post["geocent_time"].values, log_likelihood=post["log_likelihood"].values)
    res = {"event": name, "tool": f"bilby {bilby.__version__} + dynesty", "waveform": "IMRPhenomD (aligned spin)",
           "gps_trigger_h1": gps_trig, "segment_gps": [gps_start, gps_start + DUR], "f_range_hz": [F_MIN, F_MAX],
           "nlive": 500, "naccept": 30, "npool": npool, "sampling_hours_this_run": hours,
           "sampling_time_s_total": float(result.sampling_time.total_seconds()) if hasattr(result.sampling_time, "total_seconds") else float(result.sampling_time),
           "n_samples": int(len(post)), "n_likelihood_evaluations": int(result.num_likelihood_evaluations),
           "log_bayes_factor": float(result.log_bayes_factor), "log_evidence_err": float(result.log_evidence_err),
           "max_log_likelihood_ratio": float(np.max(post["log_likelihood"].values)),
           "mc_det": q90(mc), "q": q90(q), "m1_det": q90(m1), "m2_det": q90(m2), "mtot_det": q90(m1 + m2),
           "chi_eff": q90(chieff), "chi_1": q90(c1), "chi_2": q90(c2), "p_chi_eff_positive": float(np.mean(chieff > 0)),
           "luminosity_distance_mpc": q90(dl), "z": q90(z), "mc_source": q90(mc / (1 + z)),
           "m1_source": q90(m1 / (1 + z)), "m2_source": q90(m2 / (1 + z)),
           "dt_h_minus_l_ms": None}
    # arrival-time difference implied by the sky posterior
    try:
        dts = []
        for k in range(0, len(post), max(1, len(post) // 2000)):
            r = post.iloc[k]
            dts.append(ifos[0].time_delay_from_geocenter(r["ra"], r["dec"], r["geocent_time"])
                       - ifos[1].time_delay_from_geocenter(r["ra"], r["dec"], r["geocent_time"]))
        res["dt_h_minus_l_ms"] = q90(np.array(dts) * 1e3)
    except Exception as exc:  # pragma: no cover
        res["dt_error"] = str(exc)
    json.dump(res, open(ROOT / "results" / f"pe_bilby_{name}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    ap.add_argument("--npool", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    main(a.event, a.npool, a.dry)
