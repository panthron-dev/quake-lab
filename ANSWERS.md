# Day 1 Lab: Answers

Name: Muhammad Nouman Jaffar

All numbers come from the USGS `all_week` feed generated 2026-10-22 05:35 UTC
(1942 raw rows → 1894 earthquakes after cleaning). Reproduce them with
`python scripts/experiments.py` (writes `reports/experiments.json`).

## 1. Leakage
Which columns did you drop and why? Is `tsunami` leaky? Is `magType`?

I dropped `title`, `sig`, `mmi`, `cdi`, `felt`, `alert`, `tsunami`, `magType` and `types`
(see `config.LEAKY`), and `mag` itself is removed from X because `big_quake` is defined from it.
The test for each column was *"would I know this at the moment I predict?"* `title` contains
the magnitude as text, `sig` is computed from it, and `mmi`/`cdi`/`felt`/`alert` only exist
after a quake is big enough to be felt or to trigger ShakeMap/PAGER.

- **`tsunami`: yes, leaky.** Only 2 of 1894 events had `tsunami = 1`, and both were big quakes
  (100 %), against a 6.2 % base rate. The flag is set *because* the quake was large and oceanic.
- **`magType`: yes, leaky.** The method is picked by size: `ml` (1281 events) was big 0.1 % of the
  time, `md` (441) 0 %, but `mb` (142) 67 % and `mww` (21) 100 %. Knowing the type nearly gives
  you the answer.

## 2. Stream vs batch
How many events changed (same `id`, newer `updated`) during your stream window? What does that
tell you about "latest version wins"?

Within one batch snapshot the count is 0: every `id` appears once (1942 rows, 1942 after
`dedupe_latest`), because the feed file itself always shows only the latest version.
The changes appear only when you poll repeatedly. In my stream window
(`python -m quakes.stream --interval 60 --duration 2400`) **___ events** came back with the same
`id` and a newer `updated` (fill in the number from `python -m quakes.run`, line
"events whose `updated` moved").

This tells me that an event record is not final when it's first published: USGS revises
magnitude, location and review status after the first automatic solution. Keeping the row
with the greatest `updated` gives the most accurate version. It also means the stream must
dedupe on `(id, updated)`, not on `id` alone, or those revisions would be thrown away.

## 3. Outliers
Your decision on negative depth and negative magnitude, with reasoning.

**I kept both, because they are real measurements, not errors.**
- **Negative depth:** 28 events (minimum −3.17 km), mostly from the volcano and mountain
  networks `av`, `hv`, `uu` and `nc`. Depth is measured from sea level, so a quake under a
  mountain or volcano can sit *above* sea level and get a negative depth.
- **Negative magnitude:** 82 events (minimum −1.26), 78 of them from `av`, the Alaska Volcano
  Observatory. Magnitude is a log scale, so M −1 is just a very small event that sensitive local
  instruments can pick up.

The IQR rule flagged 276 depths and 170 magnitudes. That's far too many to delete, and these
rows are rare, not wrong. Instead of dropping them I use `RobustScaler` (median/IQR), which
isn't dragged around by the long tail.

## 4. Cardinality
You grouped `region` to top-k. Name one alternative encoding and one risk it carries.

`region` had 77 distinct values, 31 of them seen only once; the top 15 cover 92.7 % of rows.
**Alternative: target (mean) encoding**, which replaces each region with the average
`big_quake` rate for that region. It makes one column instead of 16.
**Risk: target leakage and overfitting.** For a region seen once, the encoded value *is* that
row's label. It has to be computed out-of-fold on the training set only, with smoothing toward
the global mean. (Frequency encoding is another option. Its risk is that two very different
regions with the same count become identical to the model.)
