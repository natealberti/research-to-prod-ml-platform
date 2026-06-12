from pathlib import Path
from typing import Any
from ml_platform.validation.runtime_validator import validate_runtime

import yaml


REQUIRED_FILES = [
    "checkpoint.pt",
    "model_config.yaml",
    "preprocessing.yaml",
    "sample_input.pt",
    "expected_output.json",
]


class ValidationError(Exception):
    pass


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML in {path.name}: {e}") from e

    if not isinstance(data, dict):
        raise ValidationError(f"{path.name} must contain a YAML dictionary/object.")

    return data


def validate_required_files(package_dir: Path) -> list[str]:
    errors = []

    if not package_dir.exists():
        return [f"Package directory does not exist: {package_dir}"]

    if not package_dir.is_dir():
        return [f"Package path is not a directory: {package_dir}"]

    for filename in REQUIRED_FILES:
        if not (package_dir / filename).exists():
            errors.append(f"Missing required file: {filename}")

    return errors


def validate_model_config(config: dict[str, Any]) -> list[str]:
    errors = []

    required_top_level = ["model", "input", "output"]
    for key in required_top_level:
        if key not in config:
            errors.append(f"model_config.yaml missing top-level key: {key}")

    if "model" in config:
        model = config["model"]
        if not isinstance(model, dict):
            errors.append("model must be a dictionary.")
        else:
            for key in ["name", "framework", "adapter", "architecture", "checkpoint_format"]:
                if key not in model:
                    errors.append(f"model missing key: {key}")

            if model.get("framework") != "pytorch":
                errors.append("Only framework='pytorch' is supported in V1.")

            if model.get("checkpoint_format") != "state_dict":
                errors.append("Only checkpoint_format='state_dict' is supported in V1.")

    if "input" in config:
        input_cfg = config["input"]
        if not isinstance(input_cfg, dict):
            errors.append("input must be a dictionary.")
        else:
            if "shape" not in input_cfg:
                errors.append("input missing key: shape")
            elif not isinstance(input_cfg["shape"], list):
                errors.append("input.shape must be a list, e.g. [1, 3, 224, 224]")

    if "output" in config:
        output_cfg = config["output"]
        if not isinstance(output_cfg, dict):
            errors.append("output must be a dictionary.")
        else:
            if "shape" not in output_cfg:
                errors.append("output missing key: shape")
            elif not isinstance(output_cfg["shape"], list):
                errors.append("output.shape must be a list, e.g. [1, 1000]")

    return errors


def validate_package(package_dir: str | Path) -> dict[str, Any]:
    package_dir = Path(package_dir)

    errors = validate_required_files(package_dir)
    if errors:
        return {
            "valid": False,
            "stage": "required_files",
            "errors": errors,
        }

    try:
        model_config = load_yaml(package_dir / "model_config.yaml")
        preprocessing_config = load_yaml(package_dir / "preprocessing.yaml")
    except ValidationError as e:
        return {
            "valid": False,
            "stage": "yaml_parse",
            "errors": [str(e)],
        }

    errors.extend(validate_model_config(model_config))

    if errors:
        return {
            "valid": False,
            "stage": "config_validation",
            "errors": errors,
        }

    runtime_result = validate_runtime(package_dir, model_config)

    if not runtime_result["valid"]:
        return runtime_result

    return {
        "valid": True,
        "stage": "runtime_validation",
        "errors": [],
        "model_config": model_config,
        "preprocessing_config": preprocessing_config,
    }