# Parameters

Values used by the simulator and the framework. A slot is 5 minutes, a day has 288 slots, and day 0 of every simulated week is the calibration day. Parameters of the simulator are in `src/sim.py` and `src/layout.py`; parameters of the framework are in `src/model.py`, `src/resolve.py` and `src/analysis.py`.

## Department

Five zones with 31 bays.

| Zone | Bays | Waiting queue | Share of queued arrivals | Probability of admission |
| --- | --- | --- | --- | --- |
| Resus | 4 | no (direct arrivals) | | 0.41 |
| Main monitoring area | 12 | yes | 0.444 | 0.30 |
| Ambulance side examination rooms | 6 | yes | 0.222 | 0.15 |
| Central examination rooms | 4 | yes | 0.148 | 0.15 |
| Assessment Suite | 5 | yes | 0.186 | 0.15 |

For larger departments, `bays` multiplies the number of bays of every zone, and `zones` repeats the whole set of zones with one shared waiting queue.

## Patient flow

| Parameter | Value |
| --- | --- |
| Arrivals | Poisson in every slot, mean 76 per day, multiplied by the hourly share and by the weekday factor; 2% of arrivals go directly to Resus |
| Hourly share of arrivals (00:00 to 23:00) | .012 .009 .010 .009 .007 .020 .032 .056 .075 .084 .089 .070 .072 .071 .061 .057 .061 .044 .045 .032 .032 .019 .022 .011, normalised to sum to 1 |
| Weekday factors (Monday to Sunday) | 1.13 1.14 0.95 0.98 0.90 0.92 0.97 |
| Surge day | arrivals multiplied by 1.6 |
| Time in the bay | lognormal, median 39 slots, sigma 0.4, at least 3 slots |
| Boarding of admitted patients | added to the time in the bay, lognormal, median 11.6 slots, sigma 1.18, at most 150 slots |
| Cleaning | a vacated bay is blocked for 2 slots |
| Closures | a free bay closes with probability 0.0015 per slot, for 6 to 30 slots |
| Patients leaving the queue unseen | up to 2 per slot, each with probability 0.01 |
| Corridor patients | Poisson with mean 0.3 + 1.2 times the zone occupancy, at most 4 per zone |

## Data sources

The fault rate factor `fs` is 1 under the normal faults and 3 in the stress test.

| Source | Behaviour |
| --- | --- |
| Bay sensors | every bay and slot; dropout with probability 0.03; stuck episodes of 6 to 20 slots starting with probability 0.001 fs; occupied reported as free with probability 0.02 fs; free reported as occupied with probability fs (0.01 + 0.04 zone occupancy); additional drift of 0.012 per day in the central examination rooms |
| Hospital record | reservation lead of 0, 1 or 2 slots (probabilities 0.5, 0.3, 0.2); discharge lag of 0 to 3 slots (0.3, 0.3, 0.25, 0.15); entry in a neighbouring bay with probability 0.015 fs; very late update (8 to 20 slots) with probability 0.01 fs; update missing with probability 0.10 |
| Coordinator reports | per bay and slot with probability 0.06; the reported state is 0 to 2 slots old; copies the hospital record with probability min(0.5, 0.25 fs); wrong state with probability 0.04 fs |
| Queue counts | sensor: true count plus normal noise with standard deviation 1; hospital record: count 0 to 2 slots old plus Poisson noise with mean 0.7 |
| Zone census | every 3 slots per zone; occupied count of the previous slot plus corridor patients, with an error of one patient with probability 0.1 |

Every observation carries a fault label (valid, timing, meaning, granularity or error) that is used only for evaluation.

## Framework

| Parameter | Value |
| --- | --- |
| Decision interval and window | every 30 minutes (6 slots) over a window of 2 hours (24 slots) |
| Calibration | audited sample of 10% of the observations of day 0; weight of a source and context 100 ln((1 - p) / p), with p = (errors + 1) / (n + 2) limited to 0.01 to 0.49; the strict interpretation counts timing, meaning and granularity differences as errors, the aligned interpretation counts only errors |
| Aggregation | sum of weights, halved every 2 slots of age, over the newest 6 slots; priors from the occupancy and closure rates of day 0 |
| Recommendation threshold | probability of availability of at least 0.9 |
| Dispute | a bay is disputed when its availability under the two alternatives lies on different sides of the threshold; extra weight limit 150, margin 6 slots, 3 s per solve |
| Confirmation | at most 2 requests per decision, accuracy of the simulated coordinator 0.97, weight 3.48 |
| Allowances of the aligned interpretation | reservation lead 2 slots, discharge lag 3 slots, coordinator lag 2 slots, queue record lag 2 slots, census allowance 4, queue growth at most 8 per slot (scaled above six times the base size), queue departures at most 2 per slot |
| Local resolution | one zone with the waiting queue over a block of 6 slots with a margin of 3, 6 or 9 slots; no neighbouring zones; at most 40 rounds of the consistency check |
| Time limits | Experiment 1: 5 s per tile problem and 20 s per solve of the consistency check. Scalability: one budget per instance, 1,800 s, shared by the tile problems (each receives a fair share of the remaining time) and the consistency check (20% of the budget is kept for it); global resolution is limited to the same 1,800 s |

## Baselines

S1 statistical fusion: the aggregator above applied to all observations. S5 Bayesian filter: a hidden Markov model per bay over the 2 hour window, with bay state transitions estimated on day 0 (pseudo count 0.5) and the calibrated source reliabilities. S6 evidential fusion: Dempster's rule over the frame available, occupied, blocked, with the calibrated weights as weights of evidence.

