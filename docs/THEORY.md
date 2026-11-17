# Theory notes — gw-lab

These notes derive the formulas the code uses. Each section ends with the place in the code where the formula appears. Units are SI unless stated. `M☉` in seconds is G M☉ / c³ = 4.925 µs.

## 1. Quadrupole radiation

In linearised general relativity, far from a slowly moving source, the metric perturbation in transverse-traceless (TT) gauge is

    h_ij^TT(t, x) = (2G / c⁴ r) · d²Q_ij^TT/dt² (t − r/c),

where Q_ij = ∫ ρ (x_i x_j − δ_ij x²/3) d³x is the trace-free mass quadrupole and r is the distance. There is no monopole radiation (mass is conserved) and no dipole radiation (momentum is conserved), so the quadrupole is the leading term.

The radiated power, averaged over a few periods, is

    P = (G / 5c⁵) ⟨ d³Q_ij/dt³ · d³Q_ij/dt³ ⟩.

**Circular binary.** Two point masses m1, m2 with total mass M = m1 + m2, reduced mass μ = m1 m2 / M, separation a and orbital angular frequency ω (Kepler: ω² = GM / a³). The quadrupole varies at twice the orbital frequency, so the gravitational-wave frequency is f = ω / π. Working through the time derivatives gives

    P = (32/5) (G⁴ / c⁵) μ² M³ / a⁵.

For an observer at inclination ι (angle between the line of sight and the orbital angular momentum) the two polarisations are

    h₊(t) = (4 / r) (G ℳ / c²)^{5/3} (π f / c)^{2/3} · (1 + cos²ι)/2 · cos Φ(t),
    h×(t) = (4 / r) (G ℳ / c²)^{5/3} (π f / c)^{2/3} · cos ι · sin Φ(t),

with Φ the gravitational-wave phase and ℳ the **chirp mass**

    ℳ = (m1 m2)^{3/5} / (m1 + m2)^{1/5} = μ^{3/5} M^{2/5}.

At this order the amplitude and the frequency evolution depend on the masses only through ℳ.

A detector measures h(t) = F₊ h₊ + F× h×, where the antenna patterns F₊, F× follow from the detector tensor D = (x̂⊗x̂ − ŷ⊗ŷ)/2 of an interferometer with arm directions x̂, ŷ: F₊ = D:e₊, F× = D:e×. Averaged over the sky and polarisation, ⟨F₊² + F×²⟩ = 2/5.

*Code:* `detectors.py` (antenna patterns), `waveforms.taylorf2` (amplitude).

## 2. The chirp

The orbital energy is E = −G m1 m2 / 2a. Setting dE/dt = −P and using Kepler's law to replace a by f gives the frequency evolution

    df/dt = (96/5) π^{8/3} (G ℳ / c³)^{5/3} f^{11/3}.        (2.1)

Solving for ℳ:

    ℳ = (c³/G) [ (5/96) π^{−8/3} f^{−11/3} ḟ ]^{3/5}.        (2.2)

This is Eq. (1) of the discovery paper. Integrating (2.1) from f to infinity gives the time left before coalescence,

    t_c − t = (5/256) (π f)^{−8/3} (G ℳ / c³)^{−5/3},        (2.3)

or equivalently

    f^{−8/3}(t) = ((8π)^{8/3} / 5) (G ℳ / c³)^{5/3} (t_c − t).        (2.4)

So f^{−8/3} is a straight line in time and its slope gives ℳ. This is what `chirpfit.fit_chirp_mass` fits. For ℳ = 30 M☉, (2.3) gives 0.74 s from 20 Hz to coalescence; the signal sweeps through the detector band in well under a second.

**Redshift.** A source at redshift z has all its time scales stretched by (1 + z). The waveform depends on mass only through G m / c³, a time, so the detector sees a signal identical to that of a nearby system with masses m (1 + z). The mass measured from the waveform is the *detector-frame* mass m_det = (1 + z) m_source. The amplitude then gives the luminosity distance, and z follows from a cosmology.

**Where the Newtonian chirp stops.** (2.1) assumes slow motion. The characteristic velocity is v/c = (π G M f / c³)^{1/3}. The innermost stable circular orbit of a Schwarzschild black hole of mass M is at a = 6GM/c², where v/c = 6^{−1/2} ≈ 0.41 and

    f_ISCO = c³ / (6^{3/2} π G M) ≈ 4400 Hz · (M☉ / M).

For M = 70 M☉ this is about 62 Hz. Beyond it the two bodies plunge and merge, and the remnant rings down at its quasi-normal-mode frequency (about 250 Hz for a 62 M☉ remnant with spin 0.67). Inspiral-only templates stop at f_ISCO; inspiral-merger-ringdown (IMR) templates cover the whole signal.

**Stationary-phase approximation (TaylorF2).** Because amplitude and frequency change slowly compared with the phase, the Fourier transform of h(t) = A(t) cos Φ(t) is dominated, at each f, by the instant t_f when the instantaneous frequency equals f:

    h̃(f) ≈ A(t_f) / √(ḟ(t_f)) · exp{ i [2π f t_f − Φ(t_f) − π/4] } / 2.

With (2.1), A ∝ f^{2/3} and ḟ ∝ f^{11/3}, so |h̃(f)| ∝ f^{−7/6}:

    |h̃₊(f)| = √(5/24) π^{−2/3} (c / r) (G ℳ / c³)^{5/6} f^{−7/6}     (face-on).

The phase, with post-Newtonian corrections up to 3.5PN order in v = (π G M f / c³)^{1/3} and symmetric mass ratio ν = m1 m2 / M², is Eq. (3.18) of Buonanno, Iyer, Ochsner, Pan and Sathyaprakash (2009, PRD 80, 084043):

    ψ(f) = 2π f t_c − φ_c − π/4 + (3 / 128 ν v⁵) [ 1 + (20/9)(743/336 + 11ν/4) v² − 16π v³ + … ].

The leading term reproduces (2.3) through t(f) = (1/2π) dψ/df. Note that Buonanno et al. define the transform with the opposite sign from numpy and LAL, so the code uses h̃ = A f^{−7/6} e^{−iψ}.

*Code:* `waveforms.taylorf2_phase`, `waveforms.taylorf2`, `waveforms.f_isco`. The test `test_taylorf2_matches_lalsimulation` compares the phase with LAL's TaylorF2.

## 3. Matched filtering and why it is optimal

**Noise model.** The data are s(t) = n(t) + h(t). Assume n is stationary and Gaussian with one-sided power spectral density S(f): ⟨ñ(f) ñ*(f′)⟩ = ½ S(f) δ(f − f′). Define the inner product

    ⟨a | b⟩ = 4 Re ∫₀^∞ ã(f) b̃*(f) / S(f) df.

**Optimal linear filter.** Apply any linear filter K: ŝ = ∫ K̃*(f) s̃(f) df + c.c. The signal contribution is S = ⟨K′ | h⟩ and the noise variance is N² = ⟨K′ | K′⟩ if we write K̃ = K̃′ / S. By the Cauchy–Schwarz inequality,

    (S/N)² = ⟨K′ | h⟩² / ⟨K′ | K′⟩ ≤ ⟨h | h⟩,

with equality exactly when K′ ∝ h. The best filter is the expected signal divided by the noise spectrum: frequencies where the detector is noisy are down-weighted. The largest attainable SNR is the *optimal SNR* ρ_opt = √⟨h|h⟩.

**Likelihood view.** For Gaussian noise the probability of the data given a signal A·h is ∝ exp(−½ ⟨s − A h | s − A h⟩). The log likelihood ratio against noise only is

    ln Λ = A ⟨s|h⟩ − ½ A² ⟨h|h⟩.

Maximising over A gives ln Λ_max = ⟨s|h⟩² / (2⟨h|h⟩) = ρ²/2. By the Neyman–Pearson lemma, thresholding the likelihood ratio gives the highest detection probability at a fixed false-alarm probability. So the matched filter is not only the best *linear* filter; in Gaussian noise it is the best detection statistic for a known waveform.

**Unknown phase and time.** A change of the coalescence phase by φ multiplies h̃ by e^{iφ}. Using the template and its 90°-shifted copy, the phase-maximised statistic is the modulus of a complex number. A time shift t multiplies h̃ by e^{−2πift}, so all arrival times are evaluated at once by one inverse FFT:

    z(t) = 4 ∫₀^∞ s̃(f) h̃*(f) / S(f) · e^{2πift} df,    σ² = ⟨h|h⟩,    ρ(t) = |z(t)| / σ.

In noise alone, the real and imaginary parts of z/σ are independent unit Gaussians, so ρ² is χ² with two degrees of freedom: ⟨ρ²⟩ = 2 and P(ρ > x) = e^{−x²/2}. With a signal of optimal SNR ρ_opt, ⟨ρ²⟩ = ρ_opt² + 2.

**Template bank.** The masses are unknown, so the filter is repeated over a grid of templates. A signal that falls between grid points is recovered with SNR reduced by the *match* (the normalised inner product maximised over time and phase) between it and the nearest template.

**Why whitening shows the signal.** The raw strain is dominated by noise below about 20 Hz (seismic motion and suspension resonances) that is two to three orders of magnitude larger than the signal. Dividing s̃(f) by √S(f) makes the noise white. The matched filter can be read as: whiten the data, whiten the template, correlate. Whitening alone still leaves broadband noise from all frequencies, including those where the signal has no power; restricting to the signal band removes most of that noise variance.

**Signal-consistency (χ²) test.** Real noise has transients (glitches) that produce large ρ. Allen's test (2005, PRD 71, 062001) splits the template into p frequency bands that each contribute ρ²/p for a true signal, and measures

    χ² = (p / σ²) Σ_j | z_j − z/p |²,

which follows a χ² distribution with 2p − 2 degrees of freedom for a matching signal in Gaussian noise, and is large when the power is distributed differently. The re-weighted SNR ρ̂ = ρ / [(1 + (χ²_r)³)/2]^{1/6} for χ²_r > 1 (Eq. 6 of Abbott et al. 2016, PRD 93, 122003) down-ranks such triggers.

**Practical points.**
- S(f) is estimated from the data (Welch's method). The median over segments is used instead of the mean because it is less affected by a loud transient; it needs a known bias correction.
- 1/S(f) estimated this way has an impulse response as long as the Welch segments. Truncating the response of 1/√S to ±2 s bounds how far each data sample can influence the filter output, at the price of slightly smoothing narrow lines.
- The FFT treats the segment as periodic. A Tukey window removes the jump at the ends, and the first and last 8 s of each 32-s segment are discarded.

*Code:* `mf.inner`, `mf.snr_series`, `mf.chisq`, `mf.newsnr`, `psd.py`.

## 4. Two detectors, coincidence and time slides

**Coincidence.** A gravitational wave crosses both LIGO sites within the light travel time between them, 10 ms. A candidate requires triggers in both detectors within 15 ms (10 ms plus an allowance for timing error). Candidates are ranked by ρ̂c = √(ρ̂_H² + ρ̂_L²), which for Gaussian noise is the log-likelihood ratio of two independent detectors with phases and amplitudes maximised separately.

**The background problem.** To state how significant a candidate is we need the rate at which noise alone produces candidates at least as loud. The noise is not Gaussian, so this cannot be computed from first principles, and the detectors cannot be shielded from real signals to obtain a signal-free sample.

**Time slides.** Shift the trigger times of one detector by an offset much longer than 10 ms and repeat the coincidence step. No real signal can be coincident after the shift, so every coincidence found is accidental. If the noise in the two detectors is independent, each shift gives one more realisation of the accidental-coincidence experiment, with the same single-detector trigger rates and loudness distributions as the real data. With N_s shifts of data lasting T the effective background time is T_bg = N_s T, and the false-alarm rate of a candidate with statistic x is

    FAR(x) = n(≥ x) / T_bg.

If no background candidate is as loud, only a bound FAR < 1/T_bg can be stated.

**Limits of the method.**
- The shifts reuse the same triggers. With N_H and N_L single-detector triggers there are only N_H · N_L distinct pairs, however many shifts are made. The number of shifts is bounded by T/δ for step δ, so T_bg ≤ T²/δ: the reach grows as the square of the amount of data. With T = 4080 s and δ = 0.1 s, T_bg is about 5 years. The published 608,000 years needed 16 days of data.
- The estimate describes the noise as it was during the analysed stretch. It says nothing about rarer glitch classes that did not occur in that stretch.
- Real signals contaminate the background: a loud signal's trigger in one detector paired with noise in the other produces loud "background" candidates. Removing the candidate's own triggers gives the background appropriate for judging *other* events; keeping them gives a conservative estimate for the candidate itself.
- Noise correlated between the sites (for example magnetic disturbances of global origin) is not captured by time slides.

**Injections.** Sensitivity is measured by adding simulated signals to real data and running the same pipeline. The fraction recovered above the detection threshold as a function of distance is the efficiency ε(D). The sensitive volume is V = 4π ∫ ε(D) D² dD (Euclidean, valid when z ≪ 1), and the range is the radius of the sphere with that volume.

*Code:* `coinc.py`, `scripts/background.py`, `scripts/run_injections.py`.

## 5. Why a template bank measures the chirp mass well and the mass ratio badly

The stationary-phase phase of an inspiral, written as a series in v = (πMf)^{1/3} with M the total mass and η = m1 m2 / M² the symmetric mass ratio, is

    ψ(f) = 2πf t_c − φ_c − π/4 + (3 / 128 η v⁵) [ 1 + (20/9)(743/336 + 11η/4) v² − (16π − 4β) v³ + … ],

where β is the spin–orbit term defined below (β = 0 without spin; then this is Eq. (3.18) of [B09] used in `waveforms.taylorf2_phase`).

**Leading order.** The prefactor 3/(128 η v⁵) equals (3/128)(πℳf)^{−5/3}: at lowest order the phase depends on the masses only through the chirp mass ℳ = η^{3/5} M. Any two binaries with the same ℳ have the same phase evolution at this order, whatever their mass ratio.

**Where the mass ratio enters.** η appears first in the 1PN term, which is smaller than the leading term by v² ~ (πMf)^{2/3}. Its contribution to the phase is of order (3/128 η v⁵)(20/9)(743/336 + 11η/4) v², i.e. it scales as v⁻³ instead of v⁻⁵. Over the band it changes by far fewer radians than the leading term, so a change of η can be largely compensated by a small change of ℳ and of the reference time and phase. What the data constrain is one combination of (ℳ, η) tightly and the orthogonal combination loosely.

**Spin makes it worse.** For spins aligned with the orbital angular momentum the 1.5PN coefficient contains

    β = Σ_i [ (113/12)(m_i/M)² + (25/4) η ] χ_i  =  (113/12 − 19η/3) χ      (equal spins χ1 = χ2 = χ).

At fixed ℳ the 1.5PN term is proportional to −(16π − 4β) η^{−3/5} f^{−2/3}. A more unequal binary (smaller η) makes it more negative through η^{−3/5}; a positive aligned spin makes it less negative through β. The two changes can cancel, so a more unequal binary with larger aligned spin fits the data as well as a more equal one with less spin. This is the q–χ_eff degeneracy of low-mass binary black holes, with χ_eff = (m1 χ1 + m2 χ2)/M the best-measured spin combination, and it is the reason the published posterior of GW151226 stretches from q ≈ 0.25 to 1 along a line of nearly constant ℳ.

**How this shows up in numbers.**
- For a long inspiral (GW151226, tens of cycles in band) the leading phase runs through hundreds of radians, so ℳ is fixed to a fraction of a per cent; q is constrained only to a factor of a few, and only jointly with χ_eff.
- For a short, heavy signal (GW150914, about eight cycles) even ℳ is known only to several per cent, because the total phase is small. The merger–ringdown frequency, which scales as 1/M, then fixes the *total* mass better than the inspiral fixes η. Along a line of constant M, changing the mass ratio from 1 to 0.8 changes ℳ by only 0.7 %, well inside its uncertainty.
- The maximum of the SNR over a discrete bank is therefore almost flat along the degenerate direction, and its location is decided by noise and by the grid. In stage 2 the templates (36, 36) and (40, 32) M☉ differed by 0.025 in network SNR out of 23.9. A bank maximum is a point estimate without a width. The honest statement about q is the posterior, which integrates over the flat direction with a prior; that is what section 3 of the stage-5 pre-registration does.

**Why aligned spin had to be included.** With χ fixed to zero the sampler can only move in (ℳ, η); the posterior is then narrower than the truth and can be biased, because the spin direction of the degeneracy is closed off. For GW151226, whose effective spin is positive, a non-spinning template family loses SNR as well.

*Code:* `scripts/run_pe_bilby.py` (bilby + dynesty, aligned-spin IMRPhenomD), `scripts/run_pe_spin.py` (own profile likelihood), `scripts/compare_pe.py`.

## 6. Chirp mass from the leading-order chirp law alone

Section 2 gave df/dt = (96/5) π^{8/3} (Gℳ/c³)^{5/3} f^{11/3}. Integrating, f^{−8/3} is a straight line in time with slope −(8π)^{8/3}/5 · (Gℳ/c³)^{5/3}. Reading f(t) from individual zero crossings and fitting that line (stage 2) failed on H1: one spurious crossing made by noise breaks the chain of points.

The same law can be fitted to all the data at once. A signal that obeys it exactly has the Fourier transform

    h̃_N(f; ℳ) ∝ f^{−7/6} exp[ −i (3/128)(πGℳf/c³)^{−5/3} ],

a one-parameter family. Correlating the data with every member (a matched filter in which ℳ is the only shape parameter) and taking the ℳ of the largest SNR is the least-squares fit of the leading-order f(t) law to the whitened data. It uses no post-Newtonian term, no merger and no ringdown, so it is independent of the IMRPhenomD bank; and because it adds the signal coherently over the whole band it does not depend on any single cycle being visible.

The price is bias. The true phase has the higher-order terms of section 5 and, for a heavy system, the band reaches the merger. The estimator therefore returns a wrong ℳ in a reproducible way, and the size of the error has to be measured: simulated full signals of known ℳ are added to real noise at the SNR of the event, the estimator is run on them, and the median relation ℳ_raw = g(ℳ_true) is inverted. The scatter of the calibrated values is the uncertainty. [BP] obtained ∼37 M☉ from their zero-crossing fit against 30 M☉ from the full analysis; the same sign and size of bias is expected here.

For a long inspiral the Newtonian phase drifts away from the true phase by many radians, the Newtonian template cannot stay coherent with the signal, and most of the SNR is lost. The method is suited to short, loud signals.

*Code:* `chirpfilter.py`, `scripts/chirp_newton.py`.

## 7. Why a lighter binary is harder to find

**Amplitude and duration.** At fixed distance the Fourier amplitude scales as ℳ^{5/6} f^{−7/6}, and the time left before merger from frequency f as ℳ^{−5/3} f^{−8/3}. A lighter binary is quieter at every frequency but stays in band longer. For an inspiral that ends above the sensitive band the optimal SNR scales as ℳ^{5/6}.

**SNR is spread over many cycles.** The number of cycles spent near frequency f is N ≈ f²/(df/dt) ∝ (ℳf)^{−5/3}. The matched-filter SNR² is, roughly, the number of cycles times the squared ratio of the strain amplitude to the noise amplitude in one cycle. A heavy binary collects its SNR in a few cycles, each of which stands well above the noise after whitening, so the signal is visible in the time series and in a spectrogram. A light binary collects comparable SNR from tens of cycles, each of which is below the noise. No single cycle and no single time–frequency pixel shows it: only a filter that stays in phase with the signal over the whole inspiral adds it up.

**Phase accuracy.** Staying in phase for N cycles requires the template phase to be right to a fraction of a radian out of 2πN. This is why the higher post-Newtonian terms, and therefore the mass ratio and the spins, matter for a light binary and hardly matter for a heavy one, and why the chirp mass of a light binary is measured so much better.

**More templates, more trials.** The number of templates needed to cover a region of parameter space at a given minimal match grows roughly with the number of cycles to a power set by the number of parameters. A bank for light binaries with spin is much denser than one for heavy binaries. Every extra template is another chance for noise to produce a trigger, so the background at a given SNR is higher, while the signal SNR is lower.

**Stochastic placement.** Geometric lattices exist for non-spinning inspirals; with spin and merger a simpler approach is used. Propose a template at random; compute its match (overlap maximised over time and phase, weighted by the noise PSD) with the templates already accepted; keep it only if every match is below the minimal match. Stop when proposals are rejected many times in a row. The result is not optimal but guarantees that accepted templates are not redundant; coverage is checked afterwards with random test signals (the fitting factor).

*Code:* `bankgen.py`, `scripts/make_lowmass_bank.py`, `scripts/compare_events.py`.

## 8. Reading a time-slide background honestly

With step δ larger than twice the coincidence window w, a given pair of triggers (one per detector) can be coincident in at most one shift. The time slides therefore do nothing more than sample the set of all N_H · N_L pairs: each pair is found with probability 2w/δ. The same expected background follows from counting pairs directly. If n(≥x) pairs have combined statistic at least x, the accidental rate is

    R(x) = n(≥x) · 2w / T².

Two consequences. First, the reach of the method grows as T²: nine times more data gives 81 times more background time. Second, the tail of the distribution is made by few single-detector triggers: the loudest H1 trigger appears in up to N_L background candidates. The number of loud background candidates overstates the number of independent pieces of evidence. This can be exposed, though not removed, by (i) recomputing the rate with one file left out at a time (jackknife), which gives an error bar that respects the grouping, and (ii) building a distribution in which every trigger is used at most once, which is biased low and brackets the estimate from below.

*Code:* `coinc.py` (`all_pairs_rate`, `no_reuse`), `scripts/background5.py`.

## References

- B. P. Abbott et al., "Observation of Gravitational Waves from a Binary Black Hole Merger", PRL 116, 061102 (2016).
- B. P. Abbott et al., "GW150914: First results from the search for binary black hole coalescence", PRD 93, 122003 (2016).
- B. P. Abbott et al., "The basic physics of the binary black hole merger GW150914", Ann. Phys. 529, 1600209 (2017).
- B. Allen, "χ² time-frequency discriminator for gravitational wave detection", PRD 71, 062001 (2005).
- B. Allen et al., "FINDCHIRP: An algorithm for detection of gravitational waves from inspiraling compact binaries", PRD 85, 122006 (2012).
- A. Buonanno et al., "Comparison of post-Newtonian templates for compact binary inspiral signals in gravitational-wave detectors", PRD 80, 084043 (2009).
- S. Khan et al., "Frequency-domain gravitational waves from nonprecessing black-hole binaries. II. A phenomenological model for the advanced detector era", PRD 93, 044007 (2016) (IMRPhenomD).
- M. Maggiore, *Gravitational Waves, Volume 1: Theory and Experiments* (Oxford University Press, 2007), chapters 3, 4 and 7.
- B. P. Abbott et al., "GW151226: Observation of Gravitational Waves from a 22-Solar-Mass Binary Black Hole Coalescence", PRL 116, 241103 (2016).
- B. P. Abbott et al., "Binary Black Hole Mergers in the First Advanced LIGO Observing Run", PRX 6, 041015 (2016).
- B. P. Abbott et al., "GWTC-1: A Gravitational-Wave Transient Catalog of Compact Binary Mergers Observed by LIGO and Virgo during the First and Second Observing Runs", PRX 9, 031040 (2019).
- E. Poisson and C. M. Will, "Gravitational waves from inspiraling compact binaries: Parameter estimation using second-post-Newtonian waveforms", PRD 52, 848 (1995) (spin-orbit term β and the mass-ratio–spin degeneracy).
- I. W. Harry, B. Allen and B. S. Sathyaprakash, "Stochastic template placement algorithm for gravitational wave data analysis", PRD 80, 104014 (2009).
- G. Ashton et al., "Bilby: A user-friendly Bayesian inference library for gravitational-wave astronomy", ApJS 241, 27 (2019).
