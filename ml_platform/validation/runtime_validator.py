import json
from pathlib import Path
from typing import Any

import torch
from torchvision.models import resnet18


class RuntimeValidationError(Exception):
    pass


SUPPORTED_ARCHITECTURES = {
    "resnet18": resnet18,
}


def load_expected_output(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeValidationError("expected_output.json must contain a JSON object.")

    return data


def build_model(model_config: dict[str, Any]) -> torch.nn.Module:
    model_section = model_config["model"]

    architecture = model_section["architecture"]
    num_classes = model_section.get("num_classes")

    if architecture not in SUPPORTED_ARCHITECTURES:
        supported = sorted(SUPPORTED_ARCHITECTURES.keys())
        raise RuntimeValidationError(
            f"Unsupported architecture '{architecture}'. Supported architectures: {supported}"
        )

    if num_classes is None:
        raise RuntimeValidationError("model.num_classes is required for torchvision classifiers.")

    model_builder = SUPPORTED_ARCHITECTURES[architecture]
    model = model_builder(num_classes=int(num_classes))
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> torch.nn.Module:
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch versions may not support weights_only.
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        raise RuntimeValidationError(f"Failed to load checkpoint: {e}") from e

    if not isinstance(state_dict, dict):
        raise RuntimeValidationError("Checkpoint must be a PyTorch state_dict dictionary.")

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeValidationError(f"Checkpoint did not match model architecture: {e}") from e

    if missing_keys:
        raise RuntimeValidationError(f"Missing checkpoint keys: {missing_keys}")

    if unexpected_keys:
        raise RuntimeValidationError(f"Unexpected checkpoint keys: {unexpected_keys}")

    return model


def validate_sample_input(sample_input: Any, expected_shape: list[int]) -> torch.Tensor:
    if not isinstance(sample_input, torch.Tensor):
        raise RuntimeValidationError("sample_input.pt must contain a torch.Tensor.")

    if list(sample_input.shape) != expected_shape:
        raise RuntimeValidationError(
            f"Sample input shape mismatch. Expected {expected_shape}, got {list(sample_input.shape)}"
        )

    return sample_input


def validate_output(output: torch.Tensor, expected_output: dict[str, Any]) -> None:
    expected_shape = expected_output.get("shape")
    expected_dtype = expected_output.get("dtype")

    if expected_shape is None:
        raise RuntimeValidationError("expected_output.json missing key: shape")

    if expected_dtype is None:
        raise RuntimeValidationError("expected_output.json missing key: dtype")

    if list(output.shape) != expected_shape:
        raise RuntimeValidationError(
            f"Output shape mismatch. Expected {expected_shape}, got {list(output.shape)}"
        )

    if str(output.dtype) != expected_dtype:
        raise RuntimeValidationError(
            f"Output dtype mismatch. Expected {expected_dtype}, got {str(output.dtype)}"
        )

    if torch.isnan(output).any():
        raise RuntimeValidationError("Model output contains NaN values.")

    if torch.isinf(output).any():
        raise RuntimeValidationError("Model output contains Inf values.")


def validate_runtime(package_dir: str | Path, model_config: dict[str, Any]) -> dict[str, Any]:
    package_dir = Path(package_dir)

    try:
        expected_output = load_expected_output(package_dir / "expected_output.json")

        model = build_model(model_config)
        model = load_checkpoint(model, package_dir / "checkpoint.pt")

        try:
            sample_input = torch.load(
                package_dir / "sample_input.pt",
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            sample_input = torch.load(package_dir / "sample_input.pt", map_location="cpu")

        expected_input_shape = model_config["input"]["shape"]
        sample_input = validate_sample_input(sample_input, expected_input_shape)

        with torch.no_grad():
            output = model(sample_input)

        validate_output(output, expected_output)

    except RuntimeValidationError as e:
        return {
            "valid": False,
            "stage": "runtime_validation",
            "errors": [str(e)],
        }
    except Exception as e:
        return {
            "valid": False,
            "stage": "runtime_validation",
            "errors": [f"Unexpected runtime validation error: {e}"],
        }

    return {
        "valid": True,
        "stage": "runtime_validation",
        "errors": [],
    }