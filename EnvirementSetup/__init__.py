"""EnvirementSetup package: resilient environment setup with 5x LLM retry and latest-version fallback."""

from .env_setup import (
     AttemptRecord,
    EnvironmentSetupManager,
    EnvironmentSetupResult,
    MAX_SETUP_ATTEMPTS,
    build_env_repair_prompt,
    build_latest_fallback_environment,
    collect_all_required_modules,
    extract_setup_error_summary,
    is_setup_failure,
    repair_environment_with_model,
    run_with_env_retry,
    setup_environment_with_retry,
    strip_version_specifiers,
    unpin_repo_manifests,
)

__all__ = [
    "MAX_SETUP_ATTEMPTS",
    "AttemptRecord",
    "EnvironmentSetupManager",
    "EnvironmentSetupResult",
    "build_env_repair_prompt",
    "build_latest_fallback_environment",
    "collect_all_required_modules",
    "extract_setup_error_summary",
    "is_setup_failure",
    "repair_environment_with_model",
    "run_with_env_retry",
    "setup_environment_with_retry",
    "strip_version_specifiers",
    "unpin_repo_manifests",
]
