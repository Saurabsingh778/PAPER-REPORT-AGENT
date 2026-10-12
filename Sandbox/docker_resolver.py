"""Select a Docker base whose Python runtime matches a validated Environment."""
from __future__ import annotations


def _major_minor(version: str | None, default: str = "3.10") -> str:
    parts = (version or default).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else default


def resolve_docker_image(env, gpu_available: bool = True) -> str:
    """Return an image compatible with the requested Python/framework contract.

    ``docker_image`` is an explicit escape hatch.  All generic images use the exact
    requested Python major/minor; remaining packages are installed by the runner.
    """
    if env.docker_image:
        return env.docker_image

    py_ver = _major_minor(env.language_version)
    frameworks = {framework.name.lower(): framework for framework in env.frameworks}
    torch = frameworks.get("pytorch") or frameworks.get("torch")

    if torch and (torch.version or "").startswith(("0.", "1.0", "1.1")) and gpu_available:
        # Python 3.6/CUDA 9 is the last broadly usable CUDA base for PyTorch 0.x.
        # The resolved pip requirement still pins the exact requested release.
        if py_ver == "3.6":
            return "pytorch/pytorch:0.4.1-cuda9-cudnn7-devel"

    # Avoid downloading a multi-hundred-MB PyTorch wheel when a matching CUDA runtime
    # image already supplies it.  Keep this mapping explicit: unknown combinations use
    # a Python base rather than silently changing the requested interpreter.
    modern_torch_images = {
        ("2.2", "3.10"): "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        ("2.3", "3.10"): "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
    }
    if torch and gpu_available:
        version = ".".join((torch.version or "").split(".")[:2])
        image = modern_torch_images.get((version, py_ver))
        if image:
            return image

    # Prefer a generic base to a newer framework image: this preserves the requested
    # interpreter version instead of silently upgrading an old repository.
    return f"python:{py_ver}-slim"
