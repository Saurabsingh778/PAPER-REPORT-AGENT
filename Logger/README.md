# Logger

Unified, hierarchical logging for the whole pipeline. Import everything from `Logger`.

```
Logs/runs/<run_id>/
  main.log / main.jsonl      one summary line per call + phase/session boundaries + all warnings/errors
  manifest.json              run metadata, upstream run, phase + session index
  summary.md                 readable report written when the run ends
  phases/<NN>_<name>/        RAW output of each phase
      phase.log              every log line + captured print() output, timestamped
      events.jsonl           structured events; full args/results of every call
      artifacts/             prompts, raw model responses, copied files
  local_session/             local Docker layer, logged separately
      session.log, docker/container_stdout.log, results/steps/*, raw/* (untruncated per-command output)
Logs/index.jsonl, Logs/links.json   cross-run index (PDF read -> extraction -> reproduction chain)
```

Browse: `python -m Logger.cli list | show [RUN] | phase RUN NAME | session RUN | chain RUN`
Verify: `python -m Logger.selftest` (no Docker or API keys needed)
Config: env `AGENT_LOG_ROOT`, `AGENT_LOG_REDACT`, `AGENT_LOG_CAPTURE`, `AGENT_LOG_ECHO`.
Secrets (API keys/tokens from env + common token shapes) are redacted in every file.
