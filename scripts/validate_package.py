from pathlib import Path
from pprint import pprint

from ml_platform.validation.package_validator import validate_package


PACKAGE_DIR = Path("examples/resnet18_classifier")


if __name__ == "__main__":
    result = validate_package(PACKAGE_DIR)
    pprint(result)