# AFML Research Lab

This repository is a personal research implementation of selected ideas from
*Advances in Financial Machine Learning*.

The first milestone focuses on chapters 2 and 3:

- Data structures: standard bars and information-driven bars.
- Multi-product series: ETF trick, PCA weights, and single future roll.
- Event-based sampling: symmetric CUSUM filter.
- Labeling: dynamic thresholds and the triple-barrier method.

The goal is not to build a complete production trading system. The goal is to
make the research pipeline explicit, testable, and easy to extend.

## Project Principles

- Research first, production later.
- Keep each AFML concept isolated enough to study independently.
- Prefer clear data contracts over clever abstractions.
- Make intermediate outputs inspectable in notebooks.
- Keep implementation deterministic where possible.
- Do not modify the `qlib` conda environment without explicit approval.
- Before implementing a concrete AFML function, ask whether there is an
  existing snippet or preferred reference implementation to adapt.

## Pipeline

```text
raw market data
-> bars
-> volatility target
-> CUSUM events
-> triple-barrier labels
-> research dataset
```

The first working version should be able to run this pipeline on one symbol or
one synthetic dataset before expanding to multi-product workflows.

## Development Environment

Use the existing conda environment named `qlib`.

Do not install, update, pin, or remove packages in that environment unless this
is explicitly approved.

```powershell
conda activate qlib
```

If a missing dependency blocks implementation or testing, document it first and
ask before changing the environment.

## Project Layout

```text
data/                  Local data files, ignored by git.
notebooks/             Exploratory notebooks.
src/afml/              Research library code.
tests/                 Focused unit tests.
configs/               Reproducible experiment settings.
scripts/               Command-line research workflows.
```

## Data Layout

```text
data/
+-- raw/                Original market data. Ignored by git.
+-- interim/            Cleaned or normalized data. Ignored by git.
+-- processed/          Bars, events, labels, and features. Ignored by git.
+-- external/           Reference data, contract metadata, calendars.
```

Expected raw tick-like data fields:

```text
timestamp
price
volume
```

Optional fields:

```text
symbol
side
bid
ask
dollar_value
```

Canonical bar output fields:

```text
timestamp
open
high
low
close
volume
dollar_value
tick_count
```

Canonical event and label output fields:

```text
event_time
t1
target
side
pt
sl
ret
label
```

## Module Roadmap

### 1. Data

Location: `src/afml/data/`

Purpose:

- Define canonical schemas.
- Load local CSV, Parquet, or qlib-style data.
- Validate required fields, timestamp order, missing values, and duplicate rows.

Initial files:

- `loaders.py`
- `schemas.py`
- `validation.py`

### 2. Bars

Location: `src/afml/bars/`

Purpose:

- Convert raw market data into more informative sampling units.
- Compare standard bars with information-driven bars.

Planned standard bars:

- Time bars
- Tick bars
- Volume bars
- Dollar bars

Planned information-driven bars:

- Tick imbalance bars
- Volume imbalance bars
- Dollar imbalance bars
- Tick runs bars
- Volume runs bars
- Dollar runs bars

Implementation order:

1. Dollar bars
2. Tick bars
3. Volume bars
4. Time bars
5. Tick imbalance bars
6. Volume and dollar imbalance bars
7. Runs bars

### 3. Multi-Product Series

Location: `src/afml/multi_product/`

Purpose:

- Handle series where a single observation depends on multiple instruments.
- Prepare research inputs for spreads, baskets, PCA portfolios, and futures.

Planned components:

- ETF trick: produce the value of one dollar invested in a spread.
- PCA weights: infer portfolio weights from the covariance structure.
- Single future roll: build a continuous futures series.

### 4. Sampling

Location: `src/afml/sampling/`

Purpose:

- Sample observations based on events rather than fixed time intervals.
- Reduce redundant observations and focus on meaningful price moves.

Planned components:

- Daily volatility estimates.
- Dynamic thresholds.
- Symmetric CUSUM filter.

Implementation order:

1. Daily volatility target
2. Symmetric CUSUM filter
3. Event table construction

### 5. Labeling

Location: `src/afml/labeling/`

Purpose:

- Assign supervised learning labels to event timestamps.
- Encode profit-taking, stop-loss, and time-out outcomes.

Planned components:

- Vertical barriers
- Profit-taking and stop-loss barriers
- Triple-barrier method
- Return calculation helpers
- Meta-labeling placeholder for later chapters

Implementation order:

1. Vertical barrier timestamps
2. Event target alignment
3. Triple-barrier path evaluation
4. Final label assignment

### 6. Features

Location: `src/afml/features/`

Purpose:

- Hold later feature engineering modules.
- Start with placeholders only; avoid implementing later chapters too early.

The current placeholder is `fractional_diff.py`, which belongs more naturally to
later AFML chapters.

## Notebook Roadmap

Notebooks are for exploration and sanity checks, not for core implementation.

Planned notebooks:

```text
01_data_exploration.ipynb
02_bars.ipynb
03_sampling.ipynb
04_labeling.ipynb
```

Each notebook should answer one concrete research question, such as:

- How different are time bars and dollar bars on the same input?
- How many events does CUSUM produce under different thresholds?
- How sensitive are triple-barrier labels to volatility targets?

## Script Roadmap

Scripts should wrap library functions into reproducible workflows.

Planned scripts:

```text
build_bars.py
run_cusum.py
run_labeling.py
```

Scripts should read settings from `configs/` and write outputs under
`data/processed/`.

## Testing Strategy

Tests should use small synthetic datasets with known expected results.

Initial test priorities:

- Bar boundaries and final partial bar behavior.
- CUSUM trigger timestamps.
- Dynamic threshold alignment.
- Triple-barrier label outcomes.
- Handling of empty inputs and missing required columns.

Avoid tests that depend on external data files unless the files are tiny and
checked into a dedicated fixture directory.

## Git Workflow

Recommended branches:

```text
master or main
dev
feature/bars
feature/sampling
feature/labeling
```

Suggested first commits:

1. `Initial project scaffold`
2. `Add market data schemas`
3. `Implement dollar bars`
4. `Implement CUSUM sampling`
5. `Implement triple-barrier labeling`

Data outputs should generally stay out of git.

## Near-Term Milestones

### Milestone 0: Scaffold

- Project directories exist.
- README and environment notes exist.
- Git repository is initialized.

### Milestone 1: Minimal Data Contract

- Define accepted input schemas.
- Add validation helpers.
- Add tiny synthetic fixtures for tests.

### Milestone 2: Standard Bars

- Implement dollar bars first.
- Add tick, volume, and time bars.
- Compare outputs in a notebook.

### Milestone 3: Event Sampling

- Implement daily volatility targets.
- Implement symmetric CUSUM filter.
- Produce event timestamps from bar close prices.

### Milestone 4: Triple-Barrier Labels

- Implement vertical barriers.
- Implement profit-taking and stop-loss evaluation.
- Generate labels from event timestamps and dynamic thresholds.

### Milestone 5: Multi-Product Tools

- Implement ETF trick.
- Implement PCA weights.
- Implement single future roll.

## Open Design Decisions

- Whether the primary internal data container should be pandas DataFrame only,
  or whether qlib data structures should be supported directly.
- Whether to keep scripts simple or introduce a small CLI later.
- Whether to use `master` or rename the default branch to `main`.

## Data Contracts

- Normalized market data uses a pandas `DatetimeIndex` named `timestamp`.
- The data layer normalizes timestamps to UTC timezone-aware values by default.
- Raw naive timestamps are interpreted as UTC unless `timezone=None` is passed
  explicitly for a local research exception.
- Duplicate tick timestamps are allowed by default, because exchange feeds can
  report multiple trades at the same time. Downstream bars may require an
  explicit duplicate-index policy before feature alignment.
