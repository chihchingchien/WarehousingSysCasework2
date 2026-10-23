"""ISyE 6202 HyperBasket Task 2(a): SKU-level autonomy days AND daily target stock.

Calendar baseline: D = ceil(max(booking deadline to next receipt AFTER the
booked receipt)); SKU D follows the effective preferred DC in each supply
scenario. Robustness R affects the cumulative-demand quantile, not D.

Target stock S[i,s,R,t] = R-quantile of sum of forecast daily demand across
D consecutive days starting on t. Forecast distributions use Task 1's
Poisson/NB2 mean and dispersion; draws are independent across forecast days.

This is a stated modeling assumption, not a unique optimization of D.
No ordering, supplier-fill, or inventory simulation is performed.
"""
from __future__ import annotations
from pathlib import Path
import math
import re
import pandas as pd
import numpy as np
import json


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(df: pd.DataFrame, aliases, required=True):
    lookup = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in lookup:
            return lookup[_norm(a)]
    # Conservative contains fallback for long descriptive column names.
    for a in aliases:
        na = _norm(a)
        for nc, original in lookup.items():
            if na and (na in nc or nc in na):
                return original
    if required:
        raise KeyError(f"Could not find any of {aliases}. Available columns: {df.columns.tolist()}")
    return None

def find_file(source: Path, exact: str) -> Path:
    direct = source / exact
    if direct.exists():
        return direct
    matches = [p for p in source.rglob('*.csv') if _norm(p.name) == _norm(exact)]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Expected {exact} under {source}; found {matches}")

def normalize_dc(value) -> str:
    s = str(value).strip().lower()
    if 'local' in s:
        return 'Local'
    if 'regional' in s:
        return 'Regional'
    if 'national' in s:
        return 'National'
    return str(value).strip()

def normalize_scenario(value) -> str:
    s = str(value).strip().lower()
    if 'fast' in s:
        return 'Fast'
    if 'medium' in s:
        return 'Medium'
    if 'slow' in s:
        return 'Slow'
    return str(value).strip()

def load_supplier_sequences(source: Path) -> dict[str, list[str]]:
    p = find_file(source, 'QF_Facility_Supplier_Sequence.csv')
    d = pd.read_csv(p)
    sku = find_col(d, ['sku_id'])
    # Long form: sku_id, sequence/order/rank, dc/supplier.
    rank = find_col(d, ['sequence', 'sequence_rank', 'rank', 'supplier_rank', 'priority'], required=False)
    dc = find_col(d, ['dc_type', 'supplier_dc', 'supplier', 'dc'], required=False)
    seqs = {}
    if rank is not None and dc is not None:
        x = d[[sku, rank, dc]].copy()
        x[rank] = pd.to_numeric(x[rank], errors='raise')
        for sid, g in x.sort_values(rank).groupby(sku):
            seqs[str(sid)] = [normalize_dc(v) for v in g[dc].tolist()]
        return seqs
    # Wide form: preferred_dc, second_dc, third_dc.
    cols = []
    for aliases in [
        ['preferred_dc', 'primary_dc', 'first_dc'],
        ['second_dc', 'secondary_dc', 'dc_2'],
        ['third_dc', 'tertiary_dc', 'dc_3'],
    ]:
        c = find_col(d, aliases, required=False)
        if c is not None:
            cols.append(c)
    if not cols:
        raise KeyError(f'Cannot identify supplier sequence columns in {p}. Columns={d.columns.tolist()}')
    for row in d[[sku] + cols].itertuples(index=False, name=None):
        sid, vals = str(row[0]), row[1:]
        seqs[sid] = [normalize_dc(v) for v in vals if pd.notna(v) and str(v).strip()]
    return seqs

def load_replenishment_calendar(source: Path) -> pd.DataFrame:
    """Expand rule-based supplier schedules into actual booking/receipt events."""
    p = find_file(source, 'QF_Facility_Replenishment_Scenarios.csv')
    rules = pd.read_csv(p)
    # Includes lead-in events and dates beyond the 112-day evaluation window.
    start = pd.Timestamp('2027-07-01')
    end = pd.Timestamp('2028-12-31')
    all_days = pd.date_range(start, end, freq='D')
    rows = []

    for rule in rules.itertuples(index=False):
        scenario = normalize_scenario(rule.scenario_id)
        dc = normalize_dc(rule.served_by_dc)
        label = str(rule.scheduled_days).strip().lower()
        anchor = (pd.Timestamp(rule.cycle_anchor_date).normalize()
                  if pd.notna(rule.cycle_anchor_date) else None)

        if label == 'every day':
            dates = all_days
        elif label == 'sunday and thursday':
            dates = all_days[all_days.dayofweek.isin([6, 3])]
        elif label == 'every monday':
            dates = all_days[all_days.dayofweek == 0]
        elif 'every other monday' in label:
            if anchor is None:
                raise ValueError(f'Missing cycle anchor for {scenario}/{dc}')
            dates = all_days[((all_days - anchor).days % 14) == 0]
        elif 'every fourth monday' in label:
            if anchor is None:
                raise ValueError(f'Missing cycle anchor for {scenario}/{dc}')
            dates = all_days[((all_days - anchor).days % 28) == 0]
        else:
            raise ValueError(f'Unknown scheduled_days value: {rule.scheduled_days}')

        for day in dates:
            booking_day = day - pd.Timedelta(days=int(rule.book_day_offset))
            booking_dt = pd.Timestamp(f'{booking_day.date()} {rule.book_time}')
            available_dt = pd.Timestamp(f'{day.date()} {rule.available_from}')
            rows.append((scenario, dc, booking_dt, available_dt))

    out = pd.DataFrame(rows, columns=[
        'scenario', 'dc_type', 'booking_datetime', 'availability_datetime'
    ]).drop_duplicates().sort_values([
        'scenario', 'dc_type', 'availability_datetime'
    ]).reset_index(drop=True)
    if (out.availability_datetime <= out.booking_datetime).any():
        raise ValueError('Availability must follow booking deadline.')
    return out

def effective_sequence(sequence: list[str], scenario: str) -> list[str]:
    seq = list(sequence)
    if scenario == 'Slow':
        # Case rule: no Local DC in Slow; Local-assigned products are supplied by Regional.
        seq = ['Regional' if x == 'Local' else x for x in seq]
    # De-duplicate while preserving order, because Local->Regional can create repeats.
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def build_calendar_maps(cal: pd.DataFrame):
    """Return schedules and the replenishment-driven protection period in days.

    At a booking deadline, inventory position is raised to cover demand until the
    NEXT receipt after the receipt associated with that booking. Therefore the
    protection period includes both supplier lead time and one replenishment cycle.
    """
    schedules = {}
    autonomy = {}
    for (sc, dc), g in cal.groupby(['scenario', 'dc_type']):
        g = g.sort_values('availability_datetime').reset_index(drop=True)
        schedules[(sc, dc)] = g
        if len(g) >= 2:
            next_avail = g.availability_datetime.shift(-1)
            # Last row: use the median inter-arrival interval to avoid dropping it.
            cycle = g.availability_datetime.diff().dropna().median()
            next_avail.iloc[-1] = g.availability_datetime.iloc[-1] + cycle
            protection = (next_avail - g.booking_datetime).dt.total_seconds() / 86400.0
        else:
            protection = (g.availability_datetime - g.booking_datetime).dt.total_seconds() / 86400.0
        autonomy[(sc, dc)] = max(1, int(math.ceil(float(protection.max()))))
    return schedules, autonomy

def load_forecast(path: Path) -> pd.DataFrame:
    """Load Task 1 daily predictive mean/SD and NB2 dispersion."""
    f = pd.read_csv(path)
    sku = find_col(f, ['sku_id'])
    day = find_col(f, ['date', 'demand_date'])
    mean = find_col(f, ['pred_mean', 'mean_units'])
    sd = find_col(f, ['pred_std', 'sd_units'])
    alpha = find_col(f, ['dispersion_alpha', 'alpha'], required=False)
    cols = [sku, day, mean, sd] + ([alpha] if alpha else [])
    out = f[cols].copy()
    out.columns = ['sku_id', 'date', 'mean', 'sd'] + (['alpha'] if alpha else [])
    out['sku_id'] = out.sku_id.astype(str)
    out['date'] = pd.to_datetime(out.date).dt.normalize()
    for col in ['mean', 'sd'] + (['alpha'] if alpha else []):
        out[col] = pd.to_numeric(out[col], errors='raise')
        if not np.isfinite(out[col].to_numpy()).all() or (out[col] < 0).any():
            raise ValueError(f'Forecast {col} must be finite and nonnegative.')
    if out.duplicated(['sku_id', 'date']).any():
        raise ValueError('Forecast has duplicate SKU/date rows.')
    if 'alpha' not in out:
        mu = out['mean'].to_numpy()
        variance = out['sd'].to_numpy() ** 2
        out['alpha'] = np.where(
            mu > 1e-12, np.maximum((variance - mu) / np.maximum(mu**2, 1e-12), 0), 0
        )
    return out.sort_values(['sku_id', 'date']).reset_index(drop=True)


def demand_draws(mu: np.ndarray, alpha: np.ndarray, reps: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Simulate Poisson/NB2 daily units for one SKU (reps x forecast days)."""
    draws = np.zeros((reps, len(mu)), dtype=np.int32)
    pois = alpha < 1e-10
    if pois.any():
        draws[:, pois] = rng.poisson(mu[pois], size=(reps, int(pois.sum())))
    nb = ~pois
    if nb.any():
        size = 1.0 / alpha[nb]
        p = size / (size + mu[nb])
        draws[:, nb] = rng.negative_binomial(
            size[None, :], p[None, :], size=(reps, int(nb.sum()))
        )
    return draws


def daily_target_quantiles(draws: np.ndarray, D: int, levels: list[float],
                           evaluation_days: int) -> dict[float, np.ndarray]:
    """Compute discrete quantiles of rolling D-day cumulative demand."""
    cumulative = np.pad(np.cumsum(draws, axis=1, dtype=np.int64),
                        ((0, 0), (1, 0)))
    starts = np.arange(evaluation_days)
    totals = cumulative[:, starts + D] - cumulative[:, starts]
    return {R: np.quantile(totals, R, axis=0, method='higher').astype(np.int64)
            for R in levels}


# ==================== USER CONFIGURATION ====================
# Paths are relative to this script, not the Terminal working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent
FORECAST_FILE = SCRIPT_DIR / 'outputs' / 'future_daily_forecast.csv'
OUTPUT_DIR = SCRIPT_DIR / 'task2a_outputs'

SERVICE_TARGETS = [0.50, 0.90, 0.99]
EVALUATION_DAYS = 112
TARGET_REPS = 1500
RANDOM_SEED = 6202
# ============================================================


def main():
    # Keep the existing analysis unchanged; replace command-line arguments
    # with editable constants defined above.
    from types import SimpleNamespace
    args = SimpleNamespace(
        source=SOURCE_DIR,
        forecast=FORECAST_FILE,
        output=OUTPUT_DIR,
        service_targets=SERVICE_TARGETS,
        evaluation_days=EVALUATION_DAYS,
        target_reps=TARGET_REPS,
        seed=RANDOM_SEED,
    )

    levels = sorted(set(args.service_targets))
    if any(not 0 < R < 1 for R in levels):
        ap.error('Robustness levels must be strictly between 0 and 1.')
    if args.evaluation_days < 1 or args.target_reps < 2:
        ap.error('--evaluation-days must be positive and --target-reps >= 2.')
    if not args.source.is_dir():
        ap.error('--source must be a directory, not a CSV file.')

    master = pd.read_csv(find_file(args.source, 'QF_Facility_SKU_Master.csv'))
    sku_col = find_col(master, ['sku_id'])
    if master[sku_col].isna().any():
        raise ValueError('SKU Master has missing SKU IDs.')
    skus = master[sku_col].astype(str)
    if skus.duplicated().any():
        raise ValueError('SKU Master has duplicate SKU IDs.')

    sequences = load_supplier_sequences(args.source)
    missing = set(skus) - set(sequences)
    if missing:
        raise ValueError(f'Missing supplier sequence for {len(missing)} SKUs: '
                         f'{sorted(missing)[:5]}')
    calendar = load_replenishment_calendar(args.source)
    _, autonomy_by_dc = build_calendar_maps(calendar)
    scenarios = sorted(calendar.scenario.unique())
    forecast = load_forecast(args.forecast)

    if set(forecast.sku_id) != set(skus):
        raise ValueError(
            f'Forecast SKUs ({forecast.sku_id.nunique()}) do not match '
            f'SKU Master ({len(skus)}). '
            f'Missing forecast: {len(set(skus)-set(forecast.sku_id))}; '
            f'extra forecast: {len(set(forecast.sku_id)-set(skus))}.'
        )

    rows = []
    sku_D = {}
    for sid in sorted(skus):
        for scenario in scenarios:
            seq = effective_sequence(sequences[sid], scenario)
            if not seq:
                raise ValueError(f'{sid}: empty effective supplier sequence for {scenario}')
            primary = seq[0]
            if (scenario, primary) not in autonomy_by_dc:
                raise ValueError(f'Missing replenishment calendar: {scenario}/{primary}')
            D = autonomy_by_dc[(scenario, primary)]
            sku_D[(sid, scenario)] = D
            for R in levels:
                rows.append(dict(sku_id=sid, scenario_id=scenario,
                                 service_target=R, primary_dc=primary,
                                 autonomy_days=D))
    autonomy = pd.DataFrame(rows)
    assert len(autonomy) == len(skus) * len(scenarios) * len(levels)
    assert not autonomy.duplicated(['sku_id', 'scenario_id', 'service_target']).any()

    start = forecast.date.min()
    eval_dates = pd.date_range(start, periods=args.evaluation_days, freq='D')
    max_D = int(autonomy.autonomy_days.max())
    required_end = start + pd.Timedelta(days=args.evaluation_days + max_D - 2)
    if forecast.date.max() < required_end:
        raise ValueError(f'Forecast must cover through {required_end.date()} '
                         f'for {args.evaluation_days} evaluation days and max D={max_D}. '
                         f'Actual last date: {forecast.date.max().date()}.')

    args.output.mkdir(parents=True, exist_ok=True)
    autonomy.to_csv(args.output / 'autonomy_targets_days.csv', index=False)
    dc_rows = [dict(scenario_id=sc, dc_type=dc, autonomy_days=D)
               for (sc, dc), D in sorted(autonomy_by_dc.items())]
    pd.DataFrame(dc_rows).to_csv(args.output / 'autonomy_by_dc.csv', index=False)
    (autonomy.groupby(['scenario_id', 'service_target', 'primary_dc', 'autonomy_days'])
     .size().reset_index(name='sku_count')
     .to_csv(args.output / 'autonomy_summary.csv', index=False))

    rng = np.random.default_rng(args.seed)
    target_path = args.output / 'daily_target_stock_units.csv'
    summary_path = args.output / 'target_stock_summary_by_sku.csv'
    target_header = True
    summary_rows = []
    target_rows = 0
    forecast_groups = dict(tuple(forecast.groupby('sku_id', sort=False)))

    for i, sid in enumerate(sorted(skus), 1):
        sf = forecast_groups[sid].sort_values('date').reset_index(drop=True)
        expected_dates = pd.date_range(start, periods=args.evaluation_days + max_D - 1)
        sf_dates = sf.date.iloc[:len(expected_dates)]
        if len(sf_dates) < len(expected_dates) or not np.array_equal(
                sf_dates.to_numpy(), expected_dates.to_numpy()):
            raise ValueError(f'{sid}: forecast dates must be contiguous starting '
                             f'{start.date()} through {expected_dates[-1].date()}.')

        draws = demand_draws(sf['mean'].to_numpy(), sf['alpha'].to_numpy(),
                             args.target_reps, rng)
        target_maps = {D: daily_target_quantiles(draws, D, levels, args.evaluation_days)
                       for D in {sku_D[(sid, sc)] for sc in scenarios}}

        chunks = []
        for scenario in scenarios:
            D = sku_D[(sid, scenario)]
            for R in levels:
                targets = target_maps[D][R]
                chunks.append(pd.DataFrame({
                    'sku_id': sid,
                    'scenario_id': scenario,
                    'service_target': R,
                    'date': eval_dates,
                    'effective_day': np.arange(args.evaluation_days),
                    'autonomy_days': D,
                    'target_stock_units': targets
                }))
                summary_rows.append(dict(
                    sku_id=sid, scenario_id=scenario, service_target=R,
                    autonomy_days=D, mean_target_stock_units=float(targets.mean()),
                    max_target_stock_units=int(targets.max()),
                    min_target_stock_units=int(targets.min())
                ))
        batch = pd.concat(chunks, ignore_index=True)
        batch.to_csv(target_path, mode='w' if target_header else 'a',
                     header=target_header, index=False)
        target_header = False
        target_rows += len(batch)
        if i % 100 == 0 or i == len(skus):
            print(f'Completed {i:,}/{len(skus):,} SKUs', flush=True)

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    expected_rows = len(skus) * len(scenarios) * len(levels) * args.evaluation_days
    assert target_rows == expected_rows

    manifest = dict(
        method='Calendar-based D; daily R-quantile of rolling D-day forecast demand',
        assumptions=['D is constant by preferred DC and supply scenario',
                     'R changes target units, not D',
                     'Poisson/NB2 forecast draws independent across days',
                     'No supplier-fill or inventory simulation in Task 2(a)'],
        sku_count=len(skus), scenarios=scenarios, robustness_levels=levels,
        evaluation_start=str(start.date()),
        evaluation_days=args.evaluation_days, max_autonomy_days=max_D,
        target_reps=args.target_reps, seed=args.seed,
        autonomy_policy_rows=len(autonomy), daily_target_rows=target_rows
    )
    (args.output / 'task2a_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    print(f'\\nAutonomy policies: {len(autonomy):,} rows')
    print(f'Daily target stock: {target_rows:,} rows')
    print(f'Output folder: {args.output}')
    print('Task 2(b) simulation NOT run.')


if __name__ == '__main__':
    main()
