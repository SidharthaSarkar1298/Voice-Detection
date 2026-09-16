import numpy as np
from config import (
    WEIGHT_ACOUSTIC, WEIGHT_PROSODY, WEIGHT_IDENTITY,
    THRESHOLD_SAFE, THRESHOLD_SUSPICIOUS, THRESHOLD_CRITICAL
)

class RiskEngine:
    """
    Dynamic Multi-Factor Temporal Risk Scoring Engine for SIH26104.
    Computes rolling threat indices and triggers enterprise security actions.
    """
    def __init__(self, ema_alpha=0.35):
        self.ema_alpha = ema_alpha
        self.rolling_risk = 0.0
        self.call_history = []
        self.transaction_context = {
            "is_high_value": False,
            "sensitive_keywords_detected": []
        }

    def reset_call_state(self):
        self.rolling_risk = 0.0
        self.call_history = []
        self.transaction_context = {
            "is_high_value": False,
            "sensitive_keywords_detected": []
        }

    def set_transaction_context(self, is_high_value=False, sensitive_keywords=None):
        self.transaction_context["is_high_value"] = is_high_value
        if sensitive_keywords:
            self.transaction_context["sensitive_keywords_detected"].extend(sensitive_keywords)

    def evaluate_step(self, acoustic_prob: float, prosody_anomaly: float, identity_mismatch: float):
        """
        Computes the instant and smoothed composite risk score.
        """
        # Weighted composite score
        instant_risk = (
            WEIGHT_ACOUSTIC * acoustic_prob +
            WEIGHT_PROSODY * prosody_anomaly +
            WEIGHT_IDENTITY * identity_mismatch
        )

        # Contextual boost if high-value transaction or urgent keywords present
        if self.transaction_context["is_high_value"]:
            instant_risk = min(1.0, instant_risk * 1.15)

        # Temporal EMA smoothing
        if not self.call_history:
            self.rolling_risk = instant_risk
        else:
            self.rolling_risk = (self.ema_alpha * instant_risk) + ((1.0 - self.ema_alpha) * self.rolling_risk)

        # Determine Threat Level & Action
        if self.rolling_risk >= THRESHOLD_CRITICAL:
            threat_level = "CRITICAL_ATTACK"
            action_code = "HALT_TRANSACTION_AND_ESCALATE"
            recommended_action = "🚨 CRITICAL: High-probability synthetic voice / identity impersonation detected! Lock transaction approval, request supervisor review, and initiate out-of-band biometric callback."
            badge_color = "#e74c3c"
        elif self.rolling_risk >= THRESHOLD_SAFE:
            threat_level = "SUSPICIOUS"
            action_code = "SECONDARY_VERIFICATION_REQUIRED"
            recommended_action = "⚠️ CAUTION: Anomalous acoustic/prosodic patterns detected. Request secondary authentication question or verbal confirmation before proceeding."
            badge_color = "#f39c12"
        else:
            threat_level = "AUTHENTIC_CALL"
            action_code = "PROCEED_NORMALLY"
            recommended_action = "✅ VERIFIED: Voice stream exhibits natural human acoustic biomarkers and biometric consistency."
            badge_color = "#2ecc71"

        step_record = {
            "step": len(self.call_history) + 1,
            "instant_risk": round(instant_risk, 3),
            "rolling_risk": round(self.rolling_risk, 3),
            "acoustic_prob": round(acoustic_prob, 3),
            "prosody_anomaly": round(prosody_anomaly, 3),
            "identity_mismatch": round(identity_mismatch, 3),
            "threat_level": threat_level,
            "action_code": action_code,
            "recommended_action": recommended_action,
            "badge_color": badge_color
        }

        self.call_history.append(step_record)
        return step_record
