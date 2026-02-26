"""
Shared path constants for all scripts.
All paths are relative to the project root (one level above src/).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODELS_DIR = os.path.join(ROOT, "results", "models")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
LOGS_DIR = os.path.join(ROOT, "logs")
