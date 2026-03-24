"""Kill zone time-window classification for ICT signal engine."""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# US Eastern offset (ET).  We use fixed UTC-5 (EST) for consistency;
# DST handling can be layered on top if needed.
ET_OFFSET = timedelta(hours=-5)
ET_TZ = timezone(ET_OFFSET)


@dataclass
class KillZoneInfo:
    """Current kill zone status."""
    active: bool = False
    name: str = ""  # "ASIAN", "LONDON", "NY", "NY_LUNCH", "PM"
    minutes_elapsed: int = 999
    first_15min: bool = False
    hour_et: int = 0
    minute_et: int = 0


# Kill zone definitions: (name, start_hour, start_min, end_hour, end_min) in ET
# Note: Asian wraps midnight — handled specially.
_KILLZONES = [
    ("ASIAN",    20,  0, 24,  0),   # 8pm - midnight ET (wraps to next day)
    ("LONDON",    2,  0,  5,  0),   # 2am - 5am ET
    ("NY",        7,  0, 10,  0),   # 7am - 10am ET
    ("NY_LUNCH", 12,  0, 13,  0),   # 12pm - 1pm ET
    ("PM",       13, 30, 16,  0),   # 1:30pm - 4pm ET
]


class KillZoneDetector:
    """Determines which ICT kill zone (if any) is currently active."""

    def __init__(self, use_dst: bool = False):
        """
        Args:
            use_dst: If True, shifts ET offset by 1 hour for daylight saving.
                     Default False uses fixed EST (UTC-5).
        """
        if use_dst:
            self._tz = timezone(timedelta(hours=-4))
        else:
            self._tz = ET_TZ
        self._log = logger.bind(component="KillZoneDetector")

    def classify(self, timestamp: float) -> KillZoneInfo:
        """Classify a Unix timestamp into a kill zone.

        Args:
            timestamp: Unix timestamp (seconds since epoch).

        Returns:
            KillZoneInfo with the active zone (if any).
        """
        dt = datetime.fromtimestamp(timestamp, tz=self._tz)
        hour = dt.hour
        minute = dt.minute
        total_minutes = hour * 60 + minute

        info = KillZoneInfo(hour_et=hour, minute_et=minute)

        for name, sh, sm, eh, em in _KILLZONES:
            start_min = sh * 60 + sm
            end_min = eh * 60 + em

            if name == "ASIAN":
                # Asian KZ spans 20:00 - 24:00 ET (same calendar day portion)
                # The 00:00-00:00 part is handled by the range 20:00-24:00 (1440)
                if total_minutes >= start_min:
                    info.active = True
                    info.name = name
                    info.minutes_elapsed = total_minutes - start_min
                    info.first_15min = info.minutes_elapsed < 15
                    return info
                continue

            if start_min <= total_minutes < end_min:
                info.active = True
                info.name = name
                info.minutes_elapsed = total_minutes - start_min
                info.first_15min = info.minutes_elapsed < 15
                return info

        # No kill zone active
        return info

    def classify_dt(self, dt: datetime) -> KillZoneInfo:
        """Classify from a datetime object (converted to ET internally)."""
        dt_et = dt.astimezone(self._tz)
        return self.classify(dt_et.timestamp())

    def is_active(self, timestamp: float) -> bool:
        """Quick check: is any kill zone active?"""
        return self.classify(timestamp).active

    def get_name(self, timestamp: float) -> str:
        """Return the active KZ name, or empty string."""
        info = self.classify(timestamp)
        return info.name if info.active else ""
