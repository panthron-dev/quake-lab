# Day 2-3 Lab: Part 2 additions

Copy these files into your Day 1 project (same folder structure). Then update your environment once:

```bash
conda env update -f environment.yml --prune
```

| File | Task |
|---|---|
| `quakes/etl.py` | A: ETL with logging |
| `quakes/models.py` | B: estimator and model factories (also edit `transform.build_preprocessor`) |
| `quakes/evaluate.py` | C: metrics |
| `quakes/train.py` | D: training command line |
| `notebooks/day2-3_part1_sklearn_basics.ipynb` | Part 1 (do this first) |

```bash
pytest tests/test_etl.py        # Task A
pytest tests/test_models.py     # Task B
pytest tests/test_evaluate.py   # Task C
python -m quakes.etl --mode batch
python -m quakes.train --task regression --model linear
python -m quakes.train --task classification --model logistic
```

Create `ANSWERS_ML.md` with your answers to the six lab questions in the handout.

## Scheduling note (write in this file)
Add the exact cron line and the Windows Task Scheduler steps that would run
`python -m quakes.etl --mode batch` every hour.

### Answer: run the batch ETL every hour

**Linux / macOS (cron).** Run `crontab -e` and add this one line (use your own paths;
`which python` inside the activated `niai-ml` env prints the python path):

```cron
0 * * * * cd /Users/you/quake-lab && /Users/you/miniconda3/envs/niai-ml/bin/python -m quakes.etl --mode batch >> logs/cron.log 2>&1
```

- `0 * * * *` = minute 0 of every hour, every day.
- cron starts with an almost-empty environment and no `conda activate`, so use the
  **absolute path to the env's python** and `cd` into the project first (so `import quakes` works).
- `>> logs/cron.log 2>&1` keeps stdout and errors; `etl.py` also writes `logs/etl.log`.
- Check it with `crontab -l`. On macOS, give `cron` Full Disk Access if the project is in Documents/Desktop.

**Windows (Task Scheduler).**
1. Open **Task Scheduler** → **Create Task…** (not "Basic Task", so you get every option).
2. **General**: name `Quake ETL hourly`; choose "Run whether user is logged on or not".
3. **Triggers** → New: *Daily*, start today at 00:00, tick **Repeat task every: 1 hour**,
   *for a duration of:* **Indefinitely** → OK.
4. **Actions** → New: *Start a program*
   - Program/script: `C:\Users\you\miniconda3\envs\niai-ml\python.exe`
   - Add arguments: `-m quakes.etl --mode batch`
   - Start in: `C:\Users\you\quake-lab`  ← required, or `import quakes` fails
5. **Settings**: tick "If the task is already running: Do not start a new instance".
6. OK, enter your password, then right-click the task → **Run** once to test, and check `logs\etl.log`.

**Where the log ends up.** `etl.py` writes every run to `logs/etl.log` inside the project
(file handler, plus the console). Under cron the console output also goes to `logs/cron.log`.

**How you'd know a run failed.**
- A crash (an exception) makes Python exit with a non-zero code. cron writes the traceback
  into `logs/cron.log`, and Task Scheduler shows it in **Last Run Result** (anything other
  than `0x0` is a failure).
- A run that "worked" but used old data logs `WARNING ... network failed ... using fallback`.
- Each successful run ends with `done: table now holds N rows`. If that line's timestamp is
  more than about an hour old, the schedule stopped. Check it with `tail -n 5 logs/etl.log`.

Alternative from a terminal:
`schtasks /Create /SC HOURLY /TN "Quake ETL" /TR "cmd /c cd /d C:\Users\you\quake-lab && C:\Users\you\miniconda3\envs\niai-ml\python.exe -m quakes.etl --mode batch"`
