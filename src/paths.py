"""Common path utilities for the project."""

from pathlib import Path

# Project root (workshop-cloud/), used to resolve relative paths consistently.
PROJECT_ROOT = Path(__file__).resolve().parent.parent