# gw-lab

A known-answer test on public data: a matched-filter search implemented directly with NumPy FFTs is run over LIGO Hanford (H1) and Livingston (L1) strain from the Gravitational Wave Open Science Center (GWOSC), to see whether it finds two black-hole mergers, GW150914 and GW151226, without being told when they happened, and whether its numbers match the published ones. This reproduces known detections; it is not a new one.

- Pre-registration (Korean): [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) for stages 0–4 (GW150914) and [docs/PREREGISTRATION_STAGE5.md](docs/PREREGISTRATION_STAGE5.md) for stage 5 (GW151226, extended background, new methods). Pass ranges and procedures were committed before the data were downloaded; changes are in the change logs of those files.
- Theory (English): [docs/THEORY.md](docs/THEORY.md). Pipeline and data notes (Korean): [docs/PIPELINE.md](docs/PIPELINE.md), [docs/DATA.md](docs/DATA.md). The Korean documents refer to internal reports that are not part of this repository.
- Masses are in the detector frame unless stated; intervals are 90%.

The question and the rules (known-answer test, ranges before data, event times withheld from the searches, failed checks reported as failures, earlier verdicts left unchanged) are mine. The code was written and run with AI coding agents under my direction; I reviewed each stage. The agents and I knew the published results beforehand; pre-registration guarantees only that procedures and ranges were fixed before this project's own numbers existed.

## What was done

1. The filtering chain in NumPy: median-Welch noise spectrum, whitening, noise-weighted inner product, SNR time series, χ² signal-consistency test, re-weighted SNR.
2. Templates: my own TaylorF2 inspiral phase (3.5PN), which agrees with LALSimulation to within 0.01 rad, and IMRPhenomD waveforms from LALSimulation.
3. Two searches that are not given the event time: 496 non-spinning templates (20–80 M☉) for GW150914, and a stochastic bank of 1,318 aligned-spin templates (chirp mass 7–13 M☉) for GW151226.
4. Background from 0.1 s time slides, first on 68 minutes and then on 10.2 hours (GW150914) and 5.1 hours (GW151226) of data, with an all-pairs estimate, a jackknife over files and a no-reuse distribution.
5. An injection test (900 simulated signals), a leading-order (Newtonian) chirp filter for the chirp mass from the frequency track, aligned-spin parameter estimation with bilby + dynesty (and my own likelihood + emcee as a second route), and a small CNN compared with the matched filter.

## Results

| | GW150914 | GW151226 |
|---|---|---|
| Time found minus catalogue time | +0.028 s | +0.049 s |
| Zero-lag coincidences in the event file | 1 (the event) | 1 (the event) |
| SNR: H1 / L1 / network | 19.69 / 13.52 / 23.89 | 10.30 / 7.49 / 12.74 |
| Published SNR | 19.5 / 13.3 / 23.6 | 10.5 / 7.9 / 13.0 |
| Arrival-time difference, L1 first | 7.09 ms (published 6.9) | 1.01 ms (published 1.1) |
| Chirp mass (bilby) | 30.78 (29.34–32.32) M☉ | 9.709 (9.646–9.772) M☉ |
| Mass ratio | 0.85 (0.66–0.98) | 0.59 (0.26–0.94) |
| Effective spin | −0.04 (−0.15 to +0.06) | +0.20 (+0.13 to +0.38) |
| Extended background | 427 years, loudest 9.75 | 107 years, loudest 9.47 |
| Event statistic; louder background | 23.12; none | 12.50; none |

**Pre-registered checks.** Stages 0–4: 13 of 15 reproduced. Stage 5: 19 of 20 reproduced. Not reproduced:

- **A8** (individual masses of GW150914 from the loudest template): the maximum sat on the equal-mass edge of the grid, 36.0 + 36.0 M☉, 0.2 M☉ outside the range for m2. The search fixes the chirp mass well and the mass ratio hardly at all.
- **A9** (chirp mass from the frequency track by zero crossings): the registered rule returned no measurement in H1.
- **E1** (stage 5, the same question with a leading-order chirp filter, uncorrected): 38.34 M☉ against a range ending at 38.0.

Stage 5 tested the questions behind A8 and A9 again with new methods under new checks; A8 and A9 stay "not reproduced". With a posterior instead of a grid point, the median masses of GW150914 are 38.5 and 32.7 M☉, inside the range A8 used (F1 reproduced). After a correction measured on 1,000 injections, the chirp filter gives 31.73 ± 0.54 M☉ (E2 reproduced). The full tables with ranges are in the two pre-registration files and in `results/`.

**CNN comparison.** A small 1-D CNN (57,841 parameters) did worse than the matched filter on the same 900 injections at the same false-alarm rate: 31 detections against 272 under the registered procedure, and 111 against 272 in a reference run with a wider coincidence window. This is one small model trained once, not a statement about CNNs in general.

Figures are in `docs/figures/`, result files in `results/`.

## Reproducing

Python 3.12.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-optional.txt
.venv/bin/python -m pytest                       # 32 unit tests, no data needed (they need lalsuite)
.venv/bin/python scripts/download_data.py        # GWOSC event files for GW150914, with SHA-256
.venv/bin/python scripts/run_search.py           # blind search over 4,096 s (about 10 min)
.venv/bin/python scripts/analyze_event.py
.venv/bin/python scripts/background.py
.venv/bin/python scripts/run_injections.py       # 900 injections (about 35 min)
.venv/bin/python scripts/analyze_injections.py
.venv/bin/python scripts/download_stage5.py      # GW151226 event files and O1 files for the background (3.55 GB)
.venv/bin/python scripts/run_search5.py --help   # stage 5 searches; order and options in docs/PIPELINE.md
```

The complete order of the stage 5 scripts (searches, backgrounds, bilby runs, CNN) with run times is in [docs/PIPELINE.md](docs/PIPELINE.md). `requirements-optional.txt` holds lalsuite (needed for IMRPhenomD and for the tests), bilby with dynesty < 3, and torch for the CNN. Raw strain, sampler output and model weights are not in this repository. I have not tested a full rebuild on a second machine, and the same data were not run through another search pipeline such as PyCBC.

## Data

All strain data come from GWOSC. The acknowledgement page (gwosc.org/acknowledgement, checked 2026-11-17) states: "Data from this web site are licensed under a Creative Commons Attribution 4.0 International License."

| Data | Identifier | Citation |
|---|---|---|
| GW150914 event files (GWTC-1-confident, v3), H1 and L1, 4 kHz | doi:10.7935/82H3-HH23 | R. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Open data from the first and second observing runs of Advanced LIGO and Advanced Virgo", SoftwareX 13 (2021) 100658 |
| GW151226 event files (GWTC-1-confident, v2) | gwosc.org/eventapi/json/GWTC-1-confident/GW151226/v2/ | same |
| O1 4 kHz strain files used for the background | doi:10.7935/K57P8W9D | same |
| GWTC-1 posterior samples (comparison only) | LIGO DCC P1800370 | B. P. Abbott et al., Phys. Rev. X 9, 031040 (2019). No separate licence statement was found on the DCC page |

Acknowledgement requested by GWOSC:

> This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. This material is based upon work supported by NSF's LIGO Laboratory which is a major facility fully funded by the National Science Foundation, as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. KAGRA is supported by Ministry of Education, Culture, Sports, Science and Technology (MEXT), Japan Society for the Promotion of Science (JSPS) in Japan; National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea; Academia Sinica (AS) and National Science and Technology Council (NSTC) in Taiwan.

No strain data are in this repository; `docs/DATA.md` and `data/meta/` list the files with their SHA-256. Result files hold quantities derived from the strain (triggers, SNRs, posterior summaries) and summary statistics of the GWTC-1 samples. Published values used for comparison are from Abbott et al. 2016 (PRL 116, 061102; PRD 93, 122003; PRL 116, 241102; PRL 116, 241103; PRX 6, 041015), cited with page numbers in the pre-registration files. Software used but not included: LALSuite (GPL-2.0-or-later), bilby, dynesty, PyTorch.

## Limits

- The false-alarm limits (fewer than one per 427 and per 107 years) are set by a few hours of data, not by the events' real significance; the published limits come from the whole observing run and are not reproduction targets. The tail of the background rests on a few loud triggers.
- The GW151226 search did not know the time but did know the masses: its bank was drawn narrowly around the known result. Its fitting factor is as low as 0.919 in places, and no injection efficiency was measured at these masses. H1 has data for about half of the event file.
- The banks are narrow, coincidences are not required to share a template, and there are no data-quality vetoes.
- The corrected chirp mass from the frequency track (E2) depends on the injection population used for the correction; that dependence and the 1.1 M☉ difference between the corrected H1 and L1 values are not in its quoted uncertainty.
- Parameter estimation uses one aligned-spin waveform model, without precession, higher modes or calibration and noise-spectrum uncertainty; the sampler was run once with one seed.
- The CNN's registered coincidence window was a design mistake (narrower than the CNN's timing scatter); the original result stands and the widened run is a reference only.
- A bug in the peak interpolation, present in stages 0–4, was found and fixed in stage 5 with its original output recorded (`logs/analyze_gw151226_before_fix.txt`). The GW150914 numbers were recomputed and are identical; the other templates of the fine bank were not recomputed.

## Licence

Code: MIT (see [LICENSE](LICENSE)). GWOSC data are CC BY 4.0 (above).
