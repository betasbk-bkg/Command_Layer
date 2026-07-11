# Webots Validation Layer

This folder contains a physics-based multi-robot validation panel for the command-layer resilience paper.

## Environment

- Webots R2025a
- Tested executable: `D:\Webots\msys64\mingw64\bin\webots.exe`
- World: `webots_validation/worlds/swarm_validation.wbt`
- Robot model: local differential-drive `SwarmBot` PROTO

## Run

```powershell
python webots_validation\run_webots_validation.py --seeds 8 --timeout 7200
```

For a quick load test:

```powershell
python webots_validation\run_webots_validation.py --smoke --seeds 1 --timeout 600
```

## Outputs

- `theory_outputs/webots_runs.csv`
- `theory_outputs/webots_condition_summary.csv`
- `theory_outputs/webots_contrasts.csv`
- `theory_outputs/webots_validation_report.md`
- `theory_outputs/fig_webots_command_safe_delivery.png`
- `theory_outputs/fig_webots_relay_tradeoff.png`

## Manuscript Role

Use this layer as physics validation for the relay over-allocation trade-off. The command-layer effect is useful but secondary and should be described as environment-dependent.
