"""M10 Dashboard — Data formatting and interactive visualization."""
from .panels import DashboardData
from .app import create_app
from .run_dashboard import run_and_launch

__all__ = ["DashboardData", "create_app", "run_and_launch"]
