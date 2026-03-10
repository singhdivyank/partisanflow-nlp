"""
Alerting module — evaluates drift metric rows against configured thresholds
and emits structured alerts.

Alert levels
------------
  WARNING  : Moderate drift detected; investigate but pipeline continues.
  CRITICAL : Significant drift; downstream consumers should be notified.

Supported channels (configured in base_config.yaml or .env):
  - Log output (always active)
  - Email via SMTP  (set ALERT_EMAIL_ENABLED=true + SMTP_* env vars)

The module is intentionally side-effect free during import — channels are
only activated when check_and_alert() is called.
"""

import os
import smtplib
from email.mime.text import MIMEText
from typing import Callable

from .helpers import Alert
from src.utils.constants import (
    COL_YEAR, COL_MODEL_NAME, COL_METRIC_NAME, COL_METRIC_VALUE,
    DRIFT_PSI_PROB1, DRIFT_KL_PROB1,
    PSI_WARNING_THRESHOLD, PSI_CRITICAL_THRESHOLD,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# Threshold registry
_THRESHOLDS: dict[str, list[tuple]] = {
    DRIFT_PSI_PROB1: [
        (
            PSI_CRITICAL_THRESHOLD,
            "CRITICAL",
            "PSI={value:.4f} exceeds critical threshold {threshold} for year={year}, "
            "model={model}. Significant distribution shift in prob_1.",
        ),
        (
            PSI_WARNING_THRESHOLD,
            "WARNING",
            "PSI={value:.4f} exceeds warning threshold {threshold} for year={year}, "
            "model={model}. Moderate distribution shift detected.",
        ),
    ],
    DRIFT_KL_PROB1: [
        (
            1.0,
            "CRITICAL",
            "KL divergence={value:.4f} is critically high for year={year}, "
            "model={model}.",
        ),
        (
            0.3,
            "WARNING",
            "KL divergence={value:.4f} is elevated for year={year}, model={model}.",
        ),
    ],
    "label_flip_rate": [
        (
            0.4,
            "CRITICAL",
            "Label flip rate={value:.4f} for year={year}, model={model}. "
            "Over 40 pct of paragraphs changed predicted party vs 1869.",
        ),
        (
            0.2,
            "WARNING",
            "Label flip rate={value:.4f} for year={year}, model={model}. "
            "20+ pct of paragraphs changed predicted party vs 1869.",
        ),
    ],
}


# Alert evaluation
def _evaluate_metric(row: dict) -> list[Alert]:
    """Check one metric row against the threshold registry."""
    name  = row.get(COL_METRIC_NAME, "")
    value = row.get(COL_METRIC_VALUE)
    year  = row.get(COL_YEAR, 0)
    model = row.get(COL_MODEL_NAME, "unknown")

    if value is None or name not in _THRESHOLDS:
        return []

    alerts = []

    # Standard high-value alerts — iterate thresholds from highest to lowest
    fired_level = None
    for threshold, level, template in sorted(
        _THRESHOLDS[name], key=lambda t: t[0], reverse=True
    ):
        if float(value) >= threshold and fired_level is None:
            msg = template.format(
                value=value, threshold=threshold, year=year, model=model
            )
            alerts.append(Alert(
                level=level, year=year, model_name=model,
                metric_name=name, metric_value=float(value),
                threshold=threshold, message=msg,
            ))
            fired_level = level
            break
    
    return alerts

# Notification channels
def _emit_log(alert: Alert) -> None:
    if alert.level == "CRITICAL":
        log.error("%s", alert)
    else:
        log.warning("%s", alert)

def _emit_email(alert: Alert) -> None:
    smtp_host  = os.getenv("SMTP_HOST", "localhost")
    smtp_port  = int(os.getenv("SMTP_PORT", "587"))
    smtp_user  = os.getenv("SMTP_USER", "")
    smtp_pass  = os.getenv("SMTP_PASS", "")
    from_addr  = os.getenv("ALERT_FROM_EMAIL", smtp_user)
    to_addrs   = os.getenv("ALERT_TO_EMAILS", "").split(",")

    if not to_addrs or not to_addrs[0]:
        log.warning("ALERT_TO_EMAILS not set — skipping email alert.")
        return

    subject = f"[{alert.level}] Drift Alert | year={alert.year} | {alert.metric_name}"
    body    = str(alert)
    msg     = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = ", ".join(to_addrs)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            s.starttls()
            if smtp_user:
                s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, to_addrs, msg.as_string())
        log.info("Email alert sent for metric=%s year=%d", alert.metric_name, alert.year)
    except Exception as exc:
        log.error("Failed to send email alert: %s", exc)

def _get_active_emitters() -> list[Callable[[Alert], None]]:
    """Return list of active notification functions based on env config."""
    emitters = [_emit_log]

    if os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true":
        emitters.append(_emit_email)

    return emitters

# Public API
def check_and_alert(metrics_rows: list[dict]) -> list[Alert]:
    """
    Evaluate all metric rows against thresholds and emit alerts via all
    configured channels.

    Parameters
    ----------
    metrics_rows : List of metric dicts (from data_drift or concept_drift).

    Returns
    -------
    List of Alert objects that were triggered (useful for tests / auditing).
    """
    
    emitters = _get_active_emitters()
    all_alerts: list[Alert] = []

    for row in metrics_rows:
        triggered = _evaluate_metric(row)
        for alert in triggered:
            all_alerts.append(alert)
            for emit in emitters:
                emit(alert)

    if all_alerts:
        log.info(
            "Alerting complete: %d alert(s) triggered (%d CRITICAL, %d WARNING).",
            len(all_alerts),
            sum(1 for a in all_alerts if a.level == "CRITICAL"),
            sum(1 for a in all_alerts if a.level == "WARNING"),
        )
    else:
        log.info("Alerting complete: no thresholds exceeded.")

    return all_alerts


def summarise_alerts(alerts: list[Alert]) -> dict:
    """
    Return a summary dict of triggered alerts — useful for logging to MLflow
    or returning from an Airflow task.
    """
    return {
        "total":    len(alerts),
        "critical": sum(1 for a in alerts if a.level == "CRITICAL"),
        "warning":  sum(1 for a in alerts if a.level == "WARNING"),
        "details":  [a.to_dict() for a in alerts],
    }
