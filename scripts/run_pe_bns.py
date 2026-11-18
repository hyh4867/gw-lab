"""GW170817 parameter estimation with bilby + dynesty and the relative-binning likelihood
(pre-registration stage 6, section 7).

Usage: python scripts/run_pe_bns.py [--npool 4] [--dry] [--tidal] [--check]
Checkpoints in results/pe_bilby/GW170817[_tidal]/ (git-ignored). Writes results/pe_bilby_GW170817[_tidal].json
and results/pe_bilby_GW170817[_tidal]_samples.npz (git-ignored).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gwlab import const, cosmo, psd, waveforms  # noqa: E402
import ev6common as E  # noqa: E402

FS = const.FS
DUR = 128
F_MIN, F_MAX = 30.0, 1024.0
SEED = 20261117
MC_RANGE = (1.18, 1.21)


def q90(x):
    a, b, c = np.percentile(x, [5, 50, 95])
    return {"median": float(b), "lo": float(a), "hi": float(c)}


def sky_area_90(ra, dec, ra0=None, dec0=None, nside_deg=0.5):
    """Area (deg^2) of the smallest set of equal-area pixels holding 90 % of the samples, and whether
    (ra0, dec0) lies in it. Pixels: nside_deg x nside_deg at the equator, uniform in ra and sin(dec)."""
    nra = int(round(360 / nside_deg))
    nz = int(round(nra / np.pi))
    pix_area = 4 * np.pi * (180 / np.pi) ** 2 / (nra * nz)
    ia = np.minimum((np.mod(ra, 2 * np.pi) / (2 * np.pi) * nra).astype(int), nra - 1)
    iz = np.minimum(((np.sin(dec) + 1) / 2 * nz).astype(int), nz - 1)
    cnt = np.bincount(ia * nz + iz, minlength=nra * nz)
    order = np.argsort(cnt)[::-1]
    cum = np.cumsum(cnt[order]) / len(ra)
    k = int(np.searchsorted(cum, 0.9)) + 1
    inside = None
    if ra0 is not None:
        p0 = min(int(np.mod(ra0, 2 * np.pi) / (2 * np.pi) * nra), nra - 1) * nz + min(int((np.sin(dec0) + 1) / 2 * nz), nz - 1)
        inside = bool(p0 in set(order[:k].tolist()))
    return k * pix_area, inside, pix_area


def main(npool, dry, tidal, check=False):
    import bilby
    from bilby.core.prior import Cosine, PowerLaw, Sine, Uniform
    from bilby.gw.prior import AlignedSpin, UniformInComponentsChirpMass, UniformInComponentsMassRatio

    label = "GW170817_tidal" if tidal else "GW170817"
    ev = json.load(open(ROOT / "results" / "event_gw170817.json"))
    vb = ev["variants"]["b_A_gated"]
    best = vb["fine"]
    gps_trig = vb["gps_h"]
    t_trig = gps_trig - E.A_START
    i_a = int(round((t_trig - (DUR - 2.0)) * FS))
    gps_start = E.A_START + i_a / FS
    ifos = bilby.gw.detector.InterferometerList([])
    gates = {}
    for det in E.DETS:
        x, g = E.load("A", det, gated=True)
        gates[det] = [E.A_START + a / FS for a, _ in g]
        w = const.PSD_SPAN * FS
        p0 = max(0, min(int(t_trig * FS) - w // 2, len(x) - w))
        fw, pw = psd.welch_median(x[p0:p0 + w], FS)
        seg = x[i_a:i_a + DUR * FS].copy()
        del x
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.minimum_frequency, ifo.maximum_frequency = F_MIN, F_MAX
        ifo.strain_data.roll_off = 0.4
        ifo.strain_data.set_from_time_domain_strain(seg, sampling_frequency=FS, duration=DUR, start_time=gps_start)
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(frequency_array=fw, psd_array=pw)
        ifos.append(ifo)

    def make_priors():
        pri = bilby.core.prior.PriorDict()
        pri["chirp_mass"] = UniformInComponentsChirpMass(*MC_RANGE, name="chirp_mass")
        pri["mass_ratio"] = UniformInComponentsMassRatio(0.125, 1.0, name="mass_ratio")
        pri["chi_1"] = AlignedSpin(a_prior=Uniform(0, 0.05), z_prior=Uniform(-1, 1), name="chi_1")
        pri["chi_2"] = AlignedSpin(a_prior=Uniform(0, 0.05), z_prior=Uniform(-1, 1), name="chi_2")
        pri["luminosity_distance"] = PowerLaw(alpha=2, minimum=5, maximum=150, name="luminosity_distance")
        pri["dec"] = Cosine(name="dec")
        pri["ra"] = Uniform(0, 2 * np.pi, name="ra", boundary="periodic")
        pri["theta_jn"] = Sine(name="theta_jn")
        pri["psi"] = Uniform(0, np.pi, name="psi", boundary="periodic")
        pri["phase"] = Uniform(0, 2 * np.pi, name="phase", boundary="periodic")
        pri["geocent_time"] = Uniform(gps_trig - 0.1, gps_trig + 0.1, name="geocent_time")
        if tidal:
            pri["lambda_1"] = Uniform(0, 5000, name="lambda_1")
            pri["lambda_2"] = Uniform(0, 5000, name="lambda_2")
        return pri

    pri = make_priors()
    approx = "IMRPhenomD_NRTidal" if tidal else "IMRPhenomD"
    wa = dict(waveform_approximant=approx, reference_frequency=30.0, minimum_frequency=F_MIN)
    if tidal:
        model = bilby.gw.source.lal_binary_neutron_star_relative_binning
        model_exact = bilby.gw.source.lal_binary_neutron_star
        conv = bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
    else:
        model = bilby.gw.source.lal_binary_black_hole_relative_binning
        model_exact = bilby.gw.source.lal_binary_black_hole
        conv = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    wg = bilby.gw.WaveformGenerator(duration=DUR, sampling_frequency=FS, start_time=gps_start,
                                    frequency_domain_source_model=model, parameter_conversion=conv, waveform_arguments=wa)
    outdir = ROOT / "results" / "pe_bilby" / label
    outdir.mkdir(parents=True, exist_ok=True)
    fid = dict(chirp_mass=best["mc"], mass_ratio=best["q"], chi_1=0.0, chi_2=0.0, luminosity_distance=50.0,
               dec=0.0, ra=3.0, theta_jn=0.5, psi=1.0, phase=0.0, geocent_time=gps_trig)
    if tidal:
        fid.update(lambda_1=300.0, lambda_2=300.0)
    if check:  # same fiducial as the finished run, no new maximisation
        fid = json.load(open(ROOT / "results" / f"pe_bilby_{label}.json"))["fiducial"]
    t0 = time.time()
    like = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
        ifos, wg, priors=pri, distance_marginalization=True, phase_marginalization=True,
        fiducial_parameters=fid, update_fiducial_parameters=not check,
        distance_marginalization_lookup_table=str(outdir / "distance_lookup.npz"))
    fid_used = {k: float(v) for k, v in like.fiducial_parameters.items() if np.isscalar(v)}
    print("fiducial after maximisation:", fid_used, f"{time.time() - t0:.0f}s", flush=True)
    if check:
        # after the run: relative binning against the exact likelihood on posterior samples
        wg2 = bilby.gw.WaveformGenerator(duration=DUR, sampling_frequency=FS, start_time=gps_start,
                                         frequency_domain_source_model=model_exact, parameter_conversion=conv,
                                         waveform_arguments=dict(waveform_approximant=approx, reference_frequency=30.0,
                                                                 minimum_frequency=F_MIN))
        ex = bilby.gw.GravitationalWaveTransient(ifos, wg2, priors=make_priors(), distance_marginalization=True,
                                                 phase_marginalization=True,
                                                 distance_marginalization_lookup_table=str(outdir / "distance_lookup.npz"))
        post = bilby.result.read_in_result(str(outdir / f"{label}_result.json")).posterior
        rng = np.random.default_rng(SEED)
        rows = []
        for i in rng.choice(len(post), 40, replace=False):
            p = {k: float(post[k].iloc[i]) for k in pri.keys() if k in post}
            p.setdefault("luminosity_distance", 50.0)
            p.setdefault("phase", 0.0)
            like.parameters.update(p)
            ex.parameters.update(p)
            a, b = float(like.log_likelihood_ratio()), float(ex.log_likelihood_ratio())
            rows.append({"stored": float(post["log_likelihood"].iloc[i]), "relative_binning": a, "exact": b,
                         "chirp_mass": p["chirp_mass"], "mass_ratio": p["mass_ratio"]})
            print(rows[-1], flush=True)
        d = np.array([r["relative_binning"] - r["exact"] for r in rows])
        s = np.array([r["relative_binning"] - r["stored"] for r in rows])
        out = {"n": len(rows), "fiducial": fid_used,
               "rb_minus_exact": {"mean": float(d.mean()), "std": float(d.std()), "min": float(d.min()), "max": float(d.max())},
               "recomputed_minus_stored_max_abs": float(np.abs(s).max()), "rows": rows}
        json.dump(out, open(ROOT / "results" / f"pe_bilby_{label}_rbcheck.json", "w"), indent=1)
        print({k: v for k, v in out.items() if k != "rows"}, flush=True)
        return
    if dry:
        wg2 = bilby.gw.WaveformGenerator(duration=DUR, sampling_frequency=FS, start_time=gps_start,
                                         frequency_domain_source_model=model_exact, parameter_conversion=conv,
                                         waveform_arguments=dict(waveform_approximant=approx, reference_frequency=30.0,
                                                                 minimum_frequency=F_MIN))
        pri2 = make_priors()
        ex = bilby.gw.GravitationalWaveTransient(ifos, wg2, priors=pri2, distance_marginalization=True,
                                                 phase_marginalization=True,
                                                 distance_marginalization_lookup_table=str(outdir / "distance_lookup.npz"))
        base = {k: fid_used[k] for k in pri.keys() if k in fid_used}
        base.setdefault("luminosity_distance", 50.0)
        base.setdefault("phase", 0.0)
        rows = []
        for dmc, dq in ((0, 0), (5e-5, 0), (-5e-5, 0), (1e-4, -0.1), (0, -0.2), (2e-4, 0)):
            p = dict(base, chirp_mass=base["chirp_mass"] + dmc, mass_ratio=max(0.3, min(1.0, base["mass_ratio"] + dq)))
            like.parameters.update(p)
            ex.parameters.update(p)
            a, b = like.log_likelihood_ratio(), ex.log_likelihood_ratio()
            rows.append({"d_chirp_mass": dmc, "d_mass_ratio": dq, "relative_binning": float(a), "exact": float(b), "difference": float(a - b)})
            print(rows[-1], flush=True)
        t1 = time.time()
        for _ in range(200):
            like.log_likelihood_ratio()
        ms = (time.time() - t1) / 200 * 1e3
        json.dump({"fiducial": fid_used, "checks": rows, "ms_per_likelihood": ms, "gates": gates,
                   "rho_net_hl_squared_over_2": best["rho_net_hl"] ** 2 / 2},
                  open(ROOT / "results" / f"pe_bilby_{label}_dry.json", "w"), indent=1)
        print("ms per relative-binning likelihood:", ms, flush=True)
        return
    t0 = time.time()
    result = bilby.run_sampler(
        like, pri, sampler="dynesty", nlive=1000, sample="acceptance-walk", naccept=60, dlogz=0.1,
        outdir=str(outdir), label=label, resume=True, check_point_delta_t=600, check_point_plot=False,
        npool=npool, seed=SEED, save="json", conversion_function=None)
    hours = (time.time() - t0) / 3600
    post = bilby.gw.conversion.generate_posterior_samples_from_marginalized_likelihood(result.posterior.copy(), like)
    mc, q = post["chirp_mass"].values, post["mass_ratio"].values
    m1, m2 = waveforms.masses_from_mc_q(mc, q)
    c1, c2 = post["chi_1"].values, post["chi_2"].values
    chieff = waveforms.chi_eff(m1, m2, c1, c2)
    dl = post["luminosity_distance"].values
    ra, dec = post["ra"].values, post["dec"].values
    extra = {}
    if tidal:
        l1, l2 = post["lambda_1"].values, post["lambda_2"].values
        lt = 16.0 / 13 * ((m1 + 12 * m2) * m1**4 * l1 + (m2 + 12 * m1) * m2**4 * l2) / (m1 + m2) ** 5
        extra = {"lambda_1": l1, "lambda_2": l2, "lambda_tilde": lt}
    np.savez(ROOT / "results" / f"pe_bilby_{label}_samples.npz", mc=mc, q=q, m1=m1, m2=m2, chi1=c1, chi2=c2,
             chieff=chieff, dl=dl, theta_jn=post["theta_jn"].values, ra=ra, dec=dec,
             geocent_time=post["geocent_time"].values, log_likelihood=post["log_likelihood"].values, **extra)
    area, inside, pix = sky_area_90(ra, dec, E.RA, E.DEC)
    res = {"event": "GW170817", "tool": f"bilby {bilby.__version__} + dynesty, relative binning",
           "waveform": approx + " (aligned spin, |chi| <= 0.05)", "data": "A (GWTC-1 release), L1 gated",
           "gates_gps": gates, "gps_trigger_h1": gps_trig, "segment_gps": [gps_start, gps_start + DUR],
           "f_range_hz": [F_MIN, F_MAX], "nlive": 1000, "naccept": 60, "npool": npool, "sampling_hours_this_run": hours,
           "sampling_time_s_total": float(result.sampling_time.total_seconds()) if hasattr(result.sampling_time, "total_seconds") else float(result.sampling_time),
           "n_samples": int(len(post)), "n_likelihood_evaluations": int(result.num_likelihood_evaluations),
           "log_bayes_factor": float(result.log_bayes_factor), "log_evidence_err": float(result.log_evidence_err),
           "max_log_likelihood_ratio": float(np.max(post["log_likelihood"].values)), "fiducial": fid_used,
           "mc_det": q90(mc), "q": q90(q), "m1_det": q90(m1), "m2_det": q90(m2), "mtot_det": q90(m1 + m2),
           "chi_eff": q90(chieff), "chi_1": q90(c1), "chi_2": q90(c2),
           "luminosity_distance_mpc": q90(dl), "theta_jn_deg": q90(np.rad2deg(post["theta_jn"].values)),
           "z_used_for_source_frame": E.Z, "mc_source": q90(mc / (1 + E.Z)), "m1_source": q90(m1 / (1 + E.Z)),
           "m2_source": q90(m2 / (1 + E.Z)), "mtot_source": q90((m1 + m2) / (1 + E.Z)),
           "z_from_distance": q90(cosmo.z_from_dl(dl)),
           "sky": {"area90_deg2": area, "pixel_deg2": pix, "counterpart_inside_90": inside,
                   "ra_deg": q90(np.rad2deg(ra)), "dec_deg": q90(np.rad2deg(dec)),
                   "counterpart_ra_dec_deg": [float(np.rad2deg(E.RA)), float(np.rad2deg(E.DEC))]}}
    if tidal:
        res["lambda_tilde"] = dict(q90(extra["lambda_tilde"]), upper90=float(np.percentile(extra["lambda_tilde"], 90)))
    try:
        dts = {"h_minus_l": [], "l_minus_v": []}
        for k in range(0, len(post), max(1, len(post) // 2000)):
            r = post.iloc[k]
            d = [i.time_delay_from_geocenter(r["ra"], r["dec"], r["geocent_time"]) for i in ifos]
            dts["h_minus_l"].append(d[0] - d[1])
            dts["l_minus_v"].append(d[1] - d[2])
        res["dt_ms"] = {k: q90(np.array(v) * 1e3) for k, v in dts.items()}
    except Exception as exc:  # pragma: no cover
        res["dt_error"] = str(exc)
    json.dump(res, open(ROOT / "results" / f"pe_bilby_{label}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npool", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--tidal", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    main(a.npool, a.dry, a.tidal, a.check)
