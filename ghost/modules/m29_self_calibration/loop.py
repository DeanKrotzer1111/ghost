"""Self-Calibration Loop — weekly automated recalibration of stop buffers and TP extensions."""
import asyncio
import structlog
from datetime import datetime

logger = structlog.get_logger()


class SelfCalibrationLoop:
    """Runs every Sunday at 10pm ET to recalibrate per-instrument parameters.

    Updates:
        - Stop sweep buffers (p75 from recent trades)
        - TP overshoot extensions (median by 2-hour session bucket)
        - Sends Telegram report with all changes
    """

    def __init__(self, db, sweep_analyzer, tp_analyzer, config, telegram):
        self.db = db
        self.sweep_analyzer = sweep_analyzer
        self.tp_analyzer = tp_analyzer
        self.config = config
        self.telegram = telegram

    async def run(self) -> str:
        lines = [
            f"GHOST WEEKLY CALIBRATION {datetime.now().strftime('%b %d, %Y')}",
            "=" * 54,
        ]

        instruments = await self.db.fetch(
            "SELECT DISTINCT instrument FROM trades WHERE entry_time >= NOW() - INTERVAL '7 days'"
        )

        sc, tc, nd = [], [], []

        for row in instruments:
            inst = row["instrument"]
            cnt = await self.db.fetchval(
                "SELECT COUNT(*) FROM trades WHERE instrument=$1 AND entry_time >= NOW() - INTERVAL '7 days'",
                inst,
            )
            if cnt < 5:
                nd.append(f"  {inst}: {cnt} trades skipped")
                continue

            # Recalibrate stop sweep buffers
            sw = await self.sweep_analyzer.compute(inst)
            if sw.sufficient_data:
                cur = self.config.get_sweep_buffer(inst)
                nw = sw.p75_sweep_ticks
                if abs(nw - cur) >= 0.5:
                    await self.config.set_sweep_buffer(inst, nw)
                    sc.append(f"  {inst}: {cur:.1f} to {nw:.1f} ticks n={sw.sample_size}")

            # Recalibrate TP extensions by session bucket
            for b in [2, 7, 9, 11, 13, 15]:
                ov = await self.tp_analyzer.get_overshoot(inst, b)
                if ov is not None:
                    cur = self.config.get_tp_extension(inst, b)
                    if abs(ov - cur) >= 1.0:
                        await self.config.set_tp_extension(inst, b, ov)
                        tc.append(f"  {inst} {b:02d}:00 ET: {cur:.1f} to {ov:.1f}")

        lines.append("STOP BUFFERS:")
        lines.extend(sc if sc else ["  No changes"])
        lines.append("TP EXTENSIONS:")
        lines.extend(tc if tc else ["  No changes"])
        if nd:
            lines.append("INSUFFICIENT DATA:")
            lines.extend(nd)
        lines.append(f"{len(sc)} stop + {len(tc)} TP changes applied. Ghost ready.")

        report = "\n".join(lines)

        if self.telegram:
            try:
                await self.telegram.send(report)
            except Exception as e:
                logger.error("telegram_failed", error=str(e))

        return report
