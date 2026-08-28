# -*- coding: utf-8 -*-
"""
config_validator.py
Formal startup configuration validation.
"""
from core.exceptions import BEST_EFFORT_EXCEPTIONS
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from core.leverage_policy import leverage_policy_violations

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None

if TYPE_CHECKING:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
else:
    try:
        from pydantic import BaseModel, ConfigDict, Field, ValidationError
    except ImportError:
        class BaseModel:
            model_config: dict[str, Any] = {}

            def __init__(self, **kwargs: Any) -> None:
                for k, v in kwargs.items():
                    setattr(self, k, v)

            @classmethod
            def model_validate(cls, data: dict[str, Any]) -> "BaseModel":
                return cls(**data)

        def ConfigDict(**kwargs: Any) -> dict[str, Any]:
            return dict(kwargs)

        def Field(
            default: Any = None,
            *,
            default_factory: Callable[[], Any] | None = None,
            ge: float | None = None,
            le: float | None = None,
            **kwargs: Any,
        ) -> Any:
            _ = (ge, le, kwargs)
            if default_factory is not None:
                return default_factory()
            return default

        class ValidationError(Exception):
            pass

log = logging.getLogger(__name__)
STRICT_MODEL_CONFIG: dict[str, Any] = ConfigDict(extra="forbid")
SAFE_MAX_ENTRY_LEVERAGE = 25


class KillSwitchSchema(BaseModel):  # type: ignore[misc]
    model_config: dict[str, Any] = STRICT_MODEL_CONFIG
    reduce_threshold: float = -0.02
    halt_threshold: float = -0.03
    stop_threshold: float = -0.05


class CircuitBreakerSchema(BaseModel):  # type: ignore[misc]
    model_config: dict[str, Any] = STRICT_MODEL_CONFIG
    error_threshold: int = Field(default=3, ge=1)
    anomaly_threshold: int = Field(default=3, ge=1)
    cooldown_seconds: int = Field(default=1800, ge=0)
    cooldown_minutes: int = Field(default=30, ge=0)
    large_loss_threshold: float = Field(default=-0.03, le=0)


class HybridWeightsSchema(BaseModel):  # type: ignore[misc]
    model_config: dict[str, Any] = STRICT_MODEL_CONFIG
    chatgpt: float = Field(default=0.35, ge=0)
    deepseek: float = Field(default=0.35, ge=0)
    transformer: float = Field(default=0.10, ge=0)
    lightgbm: float = Field(default=0.15, ge=0)
    xgboost: float = Field(default=0.0, ge=0)
    ppo_rl: float = Field(default=0.05, ge=0)
    sac_rl: float = Field(default=0.0, ge=0)
    tcn: float = Field(default=0.0, ge=0)
    ensemble: float = Field(default=0.0, ge=0)
    technical: Optional[float] = Field(default=None, ge=0)
    ai: Optional[float] = Field(default=None, ge=0)
    sentiment: Optional[float] = Field(default=None, ge=0)


class MainConfigSchema(BaseModel):  # type: ignore[misc]
    model_config: dict[str, Any] = STRICT_MODEL_CONFIG
    max_leverage: int = Field(default=SAFE_MAX_ENTRY_LEVERAGE, ge=1, le=SAFE_MAX_ENTRY_LEVERAGE)
    max_open_positions: Optional[int] = Field(default=10, ge=0, le=1000)
    trading_mode: str = "demo"
    kill_switch: KillSwitchSchema = Field(default_factory=KillSwitchSchema)
    circuit_breaker: CircuitBreakerSchema = Field(default_factory=CircuitBreakerSchema)
    hybrid_weights: HybridWeightsSchema = Field(default_factory=HybridWeightsSchema)


class RiskConfigSchema(BaseModel):  # type: ignore[misc]
    model_config: dict[str, Any] = STRICT_MODEL_CONFIG
    wallet_allocation: Dict[str, float] = Field(default_factory=dict)
    daily_loss_limit_pct: float = Field(default=0.03, ge=0, le=1)
    max_portfolio_risk_pct: float = Field(default=0.05, ge=0, le=1)
    max_leverage: int = Field(default=SAFE_MAX_ENTRY_LEVERAGE, ge=1, le=SAFE_MAX_ENTRY_LEVERAGE)


class ConfigValidator:
    """Validate runtime configuration before startup."""

    REQUIRED_FILES: list[str] = ["config.json", "config/risk.json"]
    CALIBRATION_ARTIFACT_FILES: tuple[str, ...] = (
        "calibration.json",
        "logistic_weights.json",
        "risk_schedule.json",
    )
    REQUIRED_CONFIG_PATHS: dict[str, list[tuple[str, ...]]] = {
        "max_leverage": [
            ("max_leverage",),
            ("risk", "max_leverage"),
            ("exchanges", "okx", "max_leverage"),
        ],
        "max_open_positions": [
            ("max_open_positions",),
            ("trade_parameters", "max_open_positions"),
            ("trading", "max_open_positions"),
        ],
        "kill_switch": [
            ("kill_switch",),
        ],
        "circuit_breaker": [
            ("circuit_breaker",),
        ],
        "hybrid_weights": [
            ("hybrid_weights",),
        ],
    }
    REQUIRED_RISK_CONFIG_KEYS: list[str] = ["wallet_allocation"]
    REQUIRED_ENV_VARS_LIVE: list[str] = [
        "OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE",
    ]

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_main_config_schema() -> Dict[str, Any]:
        schema_path = Path(__file__).resolve().parent / "config" / "config.schema.json"
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        if not isinstance(schema, dict):
            raise RuntimeError("config.schema.json must contain a JSON object")
        return schema

    @classmethod
    def validate_main_config_data(cls, cfg: Dict[str, Any]) -> List[str]:
        """Validate an in-memory config payload against the JSON Schema."""
        errors: List[str] = []
        if Draft202012Validator is None:
            errors.append("config.json schema validation framework error: jsonschema is not installed")
            return errors
        try:
            validator = Draft202012Validator(cls._load_main_config_schema())
            for err in sorted(validator.iter_errors(cfg), key=lambda item: list(item.absolute_path)):
                path = ".".join(str(part) for part in err.absolute_path)
                prefix = f"config.json schema[{path}]" if path else "config.json schema"
                errors.append(f"{prefix}: {err.message}")
        except BEST_EFFORT_EXCEPTIONS as exc:
            errors.append(f"config.json schema validation framework error: {exc}")
        return errors

    @classmethod
    def validate_ml_authority_data(cls, cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate only the Phase 14A ML authority contract.

        Invalid ML governance is intentionally isolated from global startup:
        callers may keep deterministic and LLM-authorized paths running while
        the universal resolver forces every ML effect to zero.
        """
        governance = cfg.get("ml_governance") if isinstance(cfg, dict) else None
        if not isinstance(governance, dict):
            return False, ["ml_governance_missing"]
        from decision.ml_governance import resolve_ml_governance

        snapshot = resolve_ml_governance(
            cfg,
            registry={"schema_version": "model-registry-v3", "models": []},
            trusted_artifacts={
                "schema_version": "artifact-trust-v2",
                "trusted_roots": {},
                "artifacts": [],
            },
            runtime_context={"runtime_mode": "paper", "lane": "audit"},
        )
        policy_errors: List[str] = []
        transformer = snapshot.models.get("transformer")
        if transformer is not None:
            policy_check = next((item for item in transformer.checks if item.name == "policy"), None)
            if policy_check is not None and not policy_check.passed:
                policy_errors.extend(part for part in policy_check.reason.split(";") if part)
        return not policy_errors, list(dict.fromkeys(policy_errors))

    @staticmethod
    def _validate_runtime_training_separation(cfg: Dict[str, Any], errors: List[str]) -> None:
        """Production runtime must not revive research/training schedulers."""
        scheduled = cfg.get("scheduled_training") if isinstance(cfg.get("scheduled_training"), dict) else {}
        runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
        services = runtime.get("services") if isinstance(runtime.get("services"), dict) else {}
        if scheduled.get("enabled") is not False:
            errors.append("scheduled_training.enabled must be false in production config")
        if services.get("scheduled_training") is not False:
            errors.append("runtime.services.scheduled_training must be false")

    @staticmethod
    def _schema_known_fields(schema_cls: type[BaseModel]) -> set[str]:
        model_fields = getattr(schema_cls, "model_fields", None)
        if isinstance(model_fields, dict):
            return set(model_fields)
        legacy_fields = getattr(schema_cls, "__fields__", None)
        if isinstance(legacy_fields, dict):
            return set(legacy_fields)
        return set()

    @classmethod
    def _validate_pydantic_subset(
        cls,
        schema_cls: type[BaseModel],
        payload: Dict[str, Any],
        source: str,
        errors: List[str],
    ) -> None:
        known_fields = cls._schema_known_fields(schema_cls)
        subset = {key: payload[key] for key in known_fields if key in payload}
        try:
            schema_cls(**subset)
        except ValidationError as exc:
            errors.append(f"{source} schema validation: {exc}")

    @classmethod
    def get_validation_errors(
        cls,
        root: Optional[Path] = None,
        skip_env: bool = False,
        config_path: Optional[Path] = None,
    ) -> List[str]:
        """Return a list of validation errors instead of just a boolean."""
        root = root or Path.cwd()
        errors: List[str] = []
        cfg: Dict[str, Any] = {}
        risk_cfg: Dict[str, Any] = {}
        trading_cfg: Dict[str, Any] = {}

        for f in cls.REQUIRED_FILES:
            fp = root / f
            if not fp.exists():
                errors.append(f"Missing required file: {f}")

        resolved_config_path = (
            Path(config_path).expanduser().resolve(strict=False)
            if config_path is not None
            else root / "config.json"
        )
        if resolved_config_path.exists():
            try:
                cfg = json.loads(resolved_config_path.read_text(encoding="utf-8"))
                if not isinstance(cfg, dict):
                    errors.append("config.json must decode to an object")
                    cfg = {}
                errors.extend(cls.validate_main_config_data(cfg))
                ml_authority_valid, ml_authority_errors = cls.validate_ml_authority_data(cfg)
                if not ml_authority_valid:
                    errors.extend(f"ml_governance warning: {item}" for item in ml_authority_errors)
                cls._check_required_config_paths(cfg, "config.json", errors)
                cls._validate_value_ranges(cfg, errors)
                cls._validate_runtime_training_separation(cfg, errors)
                cls._validate_live_readiness(cfg, root, errors)
                cls._validate_live_safety(cfg, errors)
            except json.JSONDecodeError as e:
                errors.append(f"config.json parse error: {e}")
        elif config_path is not None:
            errors.append(f"Missing required config file: {resolved_config_path}")

        risk_path = root / "config" / "risk.json"
        if risk_path.exists():
            try:
                risk_cfg = json.loads(risk_path.read_text(encoding="utf-8"))
                if not isinstance(risk_cfg, dict):
                    errors.append("config/risk.json must decode to an object")
                    risk_cfg = {}
                cls._check_required_keys(risk_cfg, cls.REQUIRED_RISK_CONFIG_KEYS,
                                         "config/risk.json", errors)
                cls._validate_risk_file_leverage_policy(risk_cfg, "config/risk.json", errors)
                cls._validate_pydantic_subset(RiskConfigSchema, risk_cfg, "risk.json", errors)
            except json.JSONDecodeError as e:
                errors.append(f"config/risk.json parse error: {e}")

        trading_path = root / "config" / "trading.json"
        if trading_path.exists():
            try:
                trading_cfg = json.loads(trading_path.read_text(encoding="utf-8"))
                if not isinstance(trading_cfg, dict):
                    errors.append("config/trading.json must decode to an object")
                    trading_cfg = {}
            except json.JSONDecodeError as e:
                errors.append(f"config/trading.json parse error: {e}")

        cls._validate_cross_file_invariants(cfg, risk_cfg, trading_cfg, errors)

        if not skip_env:
            cls._check_env_vars(errors, root, config_path=resolved_config_path)

        cls._validate_calibration_artifacts(root, errors)
            
        return errors

    @classmethod
    def validate_all(
        cls,
        root: Optional[Path] = None,
        skip_env: bool = False,
        config_path: Optional[Path] = None,
    ) -> bool:
        """Run all validation checks. Returns True if all pass."""
        root = root or Path.cwd()
        errors = cls.get_validation_errors(root, skip_env, config_path=config_path)
        warnings: List[str] = []
        fatal_errors: List[str] = []

        for err in errors:
            lower = err.lower()
            severity = cls._classify_error(lower)
            if severity == "fatal":
                fatal_errors.append(err)
            else:
                warnings.append(err)

        if warnings:
            for warn in warnings:
                if "ml_governance" in warn.lower():
                    log.warning("[ML_GOVERNANCE][BLOCKED] %s", warn)
                else:
                    log.warning("[CONFIG][WARNING] %s", warn)

        if fatal_errors:
            for err in fatal_errors:
                log.error("[CONFIG][FATAL] %s", err)
            return False

        if not warnings:
            log.info("[CONFIG] All validation checks passed")
        else:
            log.info("[CONFIG] Passed with %d warning(s)", len(warnings))
        return True

    @staticmethod
    def _classify_error(lower_msg: str) -> str:
        """Classify a validation error as 'fatal' or 'warning'.

        Fatal errors prevent startup. Warnings are logged but allow startup
        with defaults.

        Rules:
          - JSON parse errors → fatal
          - Missing safety-critical keys (kill_switch, circuit_breaker) → fatal
          - Out-of-range values on safety params → fatal
          - Missing optional keys → warning
          - Schema mismatches on non-safety params → warning
        """
        # JSON parse errors are always fatal
        if lower_msg.startswith("calibration artifact"):
            return "warning"

        if lower_msg.startswith("ml_governance warning"):
            return "warning"

        # JSON parse errors are always fatal
        if "parse error" in lower_msg:
            return "fatal"

        # Missing safety-critical config sections are fatal
        safety_keys = ("kill_switch", "circuit_breaker", "max_leverage", "live_readiness", "live_safety")
        if "missing required key" in lower_msg:
            for sk in safety_keys:
                if sk in lower_msg:
                    return "fatal"
            return "warning"

        # Out-of-range on safety parameters is fatal
        if "out of range" in lower_msg:
            for sk in safety_keys:
                if sk in lower_msg:
                    return "fatal"
            return "warning"

        # Kill switch thresholds must be negative
        if "should be negative" in lower_msg:
            return "fatal"

        # Schema validation on safety parameters
        if "schema validation" in lower_msg:
            for sk in safety_keys:
                if sk in lower_msg:
                    return "fatal"
            return "warning"

        # Missing required files
        if "missing required file" in lower_msg:
            if "app_env_file" in lower_msg:
                return "warning"
            return "fatal"

        # Everything else defaults to fatal (fail-safe)
        return "fatal"

    @staticmethod
    def _check_required_keys(data: Dict[str, Any], keys: List[str], source: str,
                             errors: List[str]) -> None:
        """Check that all required keys are present in the config dict."""
        for key in keys:
            if key not in data:
                errors.append(f"{source}: missing required key '{key}'")

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], path: Tuple[str, ...]) -> Tuple[bool, Any]:
        """Safely fetch a nested value by path."""
        cur: Any = data
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                return False, None
            cur = cur[part]
        return True, cur

    @classmethod
    def _get_first_present_value(cls, data: Dict[str, Any], paths: List[Tuple[str, ...]]) -> Any:
        """Return first present value among alternative paths."""
        for path in paths:
            present, value = cls._get_nested_value(data, path)
            if present:
                return value
        return None

    @classmethod
    def _check_required_config_paths(
        cls,
        cfg: Dict[str, Any],
        source: str,
        errors: List[str],
    ) -> None:
        """Validate required config fields with support for nested aliases."""
        for logical_key, path_options in cls.REQUIRED_CONFIG_PATHS.items():
            found = any(cls._get_nested_value(cfg, p)[0] for p in path_options)
            if not found:
                errors.append(f"{source}: missing required key '{logical_key}'")

    @classmethod
    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parent

    @classmethod
    def _get_required_env_value(cls, var: str, root: Path) -> str:
        value = os.environ.get(var, "")
        if isinstance(value, str) and value.strip():
            return value

        try:
            if root.resolve() != cls._project_root().resolve():
                return value
        except OSError:
            return value

        try:
            import settings

            ensure_bootstrap = getattr(settings, "ensure_environment_bootstrap", None)
            if callable(ensure_bootstrap):
                ensure_bootstrap()
            value = os.environ.get(var, "")
            if isinstance(value, str) and value.strip():
                return value
            attr_value = getattr(settings, var, "")
            return attr_value if isinstance(attr_value, str) else ""
        except BEST_EFFORT_EXCEPTIONS:
            return value

    @classmethod
    def _check_env_vars(
        cls,
        errors: List[str],
        root: Path,
        *,
        config_path: Optional[Path] = None,
    ) -> None:
        """Check that required exchange environment variables are set for the configured runtime."""
        resolved_config_path = config_path or (root / "config.json")
        required_vars = cls.REQUIRED_ENV_VARS_LIVE
        if resolved_config_path.exists():
            try:
                cfg = json.loads(resolved_config_path.read_text(encoding="utf-8"))
                paper_cfg = cfg.get("paper_trading", {}) if isinstance(cfg, dict) else {}
                if isinstance(paper_cfg, dict) and bool(paper_cfg.get("enabled", False)):
                    return
                exchanges = cfg.get("exchanges", {}) if isinstance(cfg, dict) else {}
                primary = "okx"
                exchange_cfg: Dict[str, Any] = {}
                if isinstance(exchanges, dict):
                    primary = str(exchanges.get("primary", "okx") or "okx").strip().lower() or "okx"
                    candidate = exchanges.get(primary, {})
                    if isinstance(candidate, dict):
                        exchange_cfg = candidate
                sandbox = bool(exchange_cfg.get("sandbox", True))
                if primary == "binance":
                    required_vars = ["BINANCE_API_KEY", "BINANCE_API_SECRET"]
                elif sandbox:
                    required_vars = ["OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"]
                else:
                    required_vars = ["OKX_LIVE_API_KEY", "OKX_LIVE_API_SECRET", "OKX_LIVE_API_PASSPHRASE"]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                required_vars = cls.REQUIRED_ENV_VARS_LIVE

        for var in required_vars:
            val = cls._get_required_env_value(var, root)
            if not val or val.strip() == "":
                errors.append(f"Missing environment variable: {var}")
            elif len(val) < 8:
                errors.append(f"Suspicious short value for {var} (len={len(val)})")

    @staticmethod
    def _validate_value_ranges(cfg: Dict[str, Any], errors: List[str]) -> None:
        """Validate that configuration values are within safe ranges."""
        for violation in leverage_policy_violations(cfg):
            errors.append(f"leverage_policy violation: {violation}")

        # Max leverage
        max_lev = ConfigValidator._get_first_present_value(
            cfg,
            [
                ("max_leverage",),
                ("risk", "max_leverage"),
                ("exchanges", "okx", "max_leverage"),
            ],
        )
        if max_lev is not None:
            if not isinstance(max_lev, (int, float)) or max_lev < 1 or max_lev > 125:
                errors.append(f"max_leverage={max_lev} out of range [1, 125]")

        # Max open positions
        max_pos = ConfigValidator._get_first_present_value(
            cfg,
            [
                ("max_open_positions",),
                ("trade_parameters", "max_open_positions"),
                ("trading", "max_open_positions"),
            ],
        )
        if max_pos is not None:
            if not isinstance(max_pos, (int, float)) or max_pos < 0 or max_pos > 1000:
                errors.append(f"max_open_positions={max_pos} out of range [0, 1000]")

        # Kill switch thresholds (should be negative)
        ks = cfg.get("kill_switch", {})
        if isinstance(ks, dict):
            for key in ("reduce_threshold", "halt_threshold", "stop_threshold"):
                val = ks.get(key)
                if val is not None and val > 0:
                    errors.append(f"kill_switch.{key}={val} should be negative")

        # Hybrid weights (should sum ≈ 1.0)
        hw = cfg.get("hybrid_weights", {})
        if isinstance(hw, dict) and hw:
            total = sum(float(v) for v in hw.values() if isinstance(v, (int, float)))
            if abs(total - 1.0) > 0.1:
                errors.append(f"hybrid_weights sum={total:.2f}, expected ≈1.0")

        # Trading mode
        mode = ConfigValidator._get_first_present_value(
            cfg,
            [("trading_mode",), ("ai_mode",)],
        )
        if mode is not None and str(mode) not in ("demo", "live", "paper", "hybrid"):
            errors.append(f"trading_mode='{mode}' not in (demo, live, paper, hybrid)")

    @staticmethod
    def _coerce_number(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            return None
        return parsed

    @classmethod
    def _present_number_values(
        cls,
        payload: Dict[str, Any],
        paths: list[tuple[str, tuple[str, ...]]],
    ) -> list[tuple[str, float]]:
        values: list[tuple[str, float]] = []
        for label, path in paths:
            present, raw = cls._get_nested_value(payload, path)
            if not present:
                continue
            parsed = cls._coerce_number(raw)
            if parsed is not None:
                values.append((label, parsed))
        return values

    @staticmethod
    def _normalized_number(value: float) -> float | int:
        if abs(value - round(value)) < 1e-9:
            return int(round(value))
        return round(value, 10)

    @classmethod
    def _assert_matching_numbers(
        cls,
        values: list[tuple[str, float]],
        logical_name: str,
        errors: List[str],
    ) -> None:
        if len(values) < 2:
            return
        normalized = {label: cls._normalized_number(value) for label, value in values}
        unique = set(normalized.values())
        if len(unique) > 1:
            errors.append(f"cross-file config conflict: {logical_name} mismatch {normalized}")

    @classmethod
    def _validate_cross_file_invariants(
        cls,
        cfg: Dict[str, Any],
        risk_cfg: Dict[str, Any],
        trading_cfg: Dict[str, Any],
        errors: List[str],
    ) -> None:
        """Validate safety-critical invariants that span config files."""
        if not isinstance(cfg, dict):
            cfg = {}
        if not isinstance(risk_cfg, dict):
            risk_cfg = {}
        if not isinstance(trading_cfg, dict):
            trading_cfg = {}

        global_leverage_values = cls._present_number_values(
            cfg,
            [
                ("config.max_leverage", ("max_leverage",)),
                ("config.risk.max_leverage", ("risk", "max_leverage")),
                ("config.exchanges.okx.max_leverage", ("exchanges", "okx", "max_leverage")),
                ("config.leverage_policy.max_entry_leverage", ("leverage_policy", "max_entry_leverage")),
            ],
        )
        global_leverage_values.extend(
            cls._present_number_values(risk_cfg, [("config/risk.max_leverage", ("max_leverage",))])
        )
        cls._assert_matching_numbers(global_leverage_values, "max_leverage", errors)
        for label, value in global_leverage_values:
            if value > SAFE_MAX_ENTRY_LEVERAGE:
                errors.append(
                    f"leverage_policy violation: {label}={cls._normalized_number(value)} "
                    f"exceeds safety cap {SAFE_MAX_ENTRY_LEVERAGE}"
                )

        for source, section in (("config.risk_profiles", cfg.get("risk_profiles")), ("config.risk_tiers", cfg.get("risk_tiers")), ("config/trading.risk_profiles", trading_cfg.get("risk_profiles")), ("config/trading.risk_tiers", trading_cfg.get("risk_tiers"))):
            if not isinstance(section, dict):
                continue
            for name, payload in section.items():
                if not isinstance(payload, dict):
                    continue
                parsed = cls._coerce_number(payload.get("max_leverage"))
                if parsed is not None and parsed > SAFE_MAX_ENTRY_LEVERAGE:
                    errors.append(
                        f"leverage_policy violation: {source}.{name}.max_leverage="
                        f"{cls._normalized_number(parsed)} exceeds safety cap {SAFE_MAX_ENTRY_LEVERAGE}"
                    )

        max_position_values = cls._present_number_values(
            cfg,
            [
                ("config.max_open_positions", ("max_open_positions",)),
                ("config.trade_parameters.max_open_positions", ("trade_parameters", "max_open_positions")),
                ("config.trading.max_open_positions", ("trading", "max_open_positions")),
            ],
        )
        max_position_values.extend(
            cls._present_number_values(
                trading_cfg,
                [
                    ("config/trading.trade_parameters.max_open_positions", ("trade_parameters", "max_open_positions")),
                    ("config/trading.trading.max_open_positions", ("trading", "max_open_positions")),
                ],
            )
        )
        cls._assert_matching_numbers(max_position_values, "max_open_positions", errors)

        cooldown_values = cls._present_number_values(
            cfg,
            [
                ("config.trade_parameters.trade_cooldown_min", ("trade_parameters", "trade_cooldown_min")),
            ],
        )
        cooldown_values.extend(
            cls._present_number_values(
                trading_cfg,
                [
                    ("config/trading.trade_parameters.trade_cooldown_min", ("trade_parameters", "trade_cooldown_min")),
                    ("config/trading.trading.trade_cooldown_min", ("trading", "trade_cooldown_min")),
                ],
            )
        )
        cls._assert_matching_numbers(cooldown_values, "trade_cooldown_minutes", errors)

        cfg_cooldowns = cls._get_nested_value(cfg, ("risk", "cooldowns", "enabled"))
        risk_cooldowns = cls._get_nested_value(risk_cfg, ("cooldowns", "enabled"))
        if cfg_cooldowns[0] and risk_cooldowns[0] and bool(cfg_cooldowns[1]) != bool(risk_cooldowns[1]):
            errors.append(
                "cross-file config conflict: risk.cooldowns.enabled mismatch "
                f"{{'config.risk.cooldowns.enabled': {bool(cfg_cooldowns[1])}, "
                f"'config/risk.cooldowns.enabled': {bool(risk_cooldowns[1])}}}"
            )

        main_accounting = cfg.get("trade_accounting") if isinstance(cfg.get("trade_accounting"), dict) else {}
        trading_accounting = (
            trading_cfg.get("trade_accounting") if isinstance(trading_cfg.get("trade_accounting"), dict) else {}
        )
        if main_accounting and trading_accounting:
            for key in ("fee_bps", "slippage_bps", "spread_bps", "funding_bps", "partial_fill_rate", "latency_ms_assumption"):
                cls._assert_matching_numbers(
                    cls._present_number_values(
                        {"main": main_accounting, "trading": trading_accounting},
                        [
                            (f"config.trade_accounting.{key}", ("main", key)),
                            (f"config/trading.trade_accounting.{key}", ("trading", key)),
                        ],
                    ),
                    f"trade_accounting.{key}",
                    errors,
                )

    @staticmethod
    def _validate_risk_file_leverage_policy(risk_cfg: Dict[str, Any], label: str, errors: List[str]) -> None:
        if not isinstance(risk_cfg, dict):
            return
        cap = SAFE_MAX_ENTRY_LEVERAGE
        values: list[tuple[str, Any]] = []
        if "max_leverage" in risk_cfg:
            values.append((f"{label}.max_leverage", risk_cfg.get("max_leverage")))
        tiers = risk_cfg.get("tiers")
        if isinstance(tiers, list):
            for idx, tier in enumerate(tiers):
                if isinstance(tier, dict) and "leverage" in tier:
                    values.append((f"{label}.tiers[{idx}].leverage", tier.get("leverage")))
        for path, value in values:
            try:
                parsed = int(float(value))
            except (TypeError, ValueError, OverflowError):
                continue
            if parsed > cap:
                errors.append(f"leverage_policy violation: {path}={value} exceeds leverage_policy.max_entry_leverage={cap}")

    @staticmethod
    def _validate_live_readiness(cfg: Dict[str, Any], root: Path, errors: List[str]) -> None:
        """Block live exchange mode until demo-forward evidence exists."""
        paper_cfg = cfg.get("paper_trading") if isinstance(cfg.get("paper_trading"), dict) else {}
        if bool(paper_cfg.get("enabled", False)):
            return

        exchanges = cfg.get("exchanges") if isinstance(cfg.get("exchanges"), dict) else {}
        primary = str(exchanges.get("primary") or "okx").strip().lower()
        exchange_cfg = exchanges.get(primary) if isinstance(exchanges.get(primary), dict) else {}
        if bool(exchange_cfg.get("sandbox", True)):
            return

        readiness = cfg.get("live_readiness") if isinstance(cfg.get("live_readiness"), dict) else {}
        if not bool(readiness.get("require_demo_before_live", True)):
            return

        min_demo_days = float(readiness.get("min_demo_days", 14) or 14)
        min_dual_write_hours = float(readiness.get("min_dual_write_hours", 48) or 48)
        min_closed_per_setup = int(readiness.get("min_closed_trades_per_setup", 50) or 50)
        required_hours = max(min_dual_write_hours, min_demo_days * 24.0)

        dual_write_hours = 0.0
        dual_write_path = root / "state" / "dual_write_events.jsonl"
        if dual_write_path.exists():
            timestamps = []
            for raw_line in dual_write_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(raw_line)
                    ts_raw = str(event.get("ts") or "").replace("Z", "+00:00")
                    if ts_raw:
                        from datetime import datetime

                        timestamps.append(datetime.fromisoformat(ts_raw))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
            if timestamps:
                dual_write_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600.0

        active_runtime_hours = 0.0
        duration_source = "active_runtime"
        active_runtime_path = root / "state" / "active_runtime_timers.json"
        if active_runtime_path.exists():
            try:
                from core.active_runtime_timer import active_runtime_status

                timer_status = active_runtime_status("demo", required_days=min_demo_days, state_file=active_runtime_path)
                active_runtime_hours = float(timer_status.get("active_elapsed_seconds") or 0.0) / 3600.0
            except BEST_EFFORT_EXCEPTIONS:
                active_runtime_hours = 0.0

        observed_hours = active_runtime_hours
        if observed_hours < required_hours:
            errors.append(
                "live_readiness.demo_duration insufficient: "
                f"{observed_hours:.2f}h observed via {duration_source}, {required_hours:.2f}h required"
            )

        approved_setups = 0
        edge_stats_path = root / "state" / "edge_stats.json"
        if edge_stats_path.exists():
            try:
                stats = json.loads(edge_stats_path.read_text(encoding="utf-8"))
                if isinstance(stats, dict):
                    for payload in stats.values():
                        if not isinstance(payload, dict):
                            continue
                        n_closed = int(payload.get("n_closed") or 0)
                        status = str(payload.get("status") or "").lower()
                        if n_closed >= min_closed_per_setup and status in {"approved", "strong"}:
                            approved_setups += 1
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                approved_setups = 0

        if approved_setups <= 0:
            errors.append(
                "live_readiness.edge_evidence insufficient: "
                f"0 approved setups with >= {min_closed_per_setup} closed demo trades"
            )

    @staticmethod
    def _validate_live_safety(cfg: Dict[str, Any], errors: List[str]) -> None:
        """Block live-mainnet side effects unless live safety gates are satisfied."""
        try:
            from core.live_safety import validate_live_safety_config

            for issue in validate_live_safety_config(cfg):
                errors.append(issue)
        except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError) as exc:
            errors.append(f"live_safety validation framework error: {exc}")

    @classmethod
    def _validate_calibration_artifacts(cls, root: Path, errors: List[str]) -> None:
        """Validate optional calibration artifacts loudly without blocking startup.

        These artifacts are allowed to be absent because several environments run
        with safe defaults. If present but corrupt, startup should still continue
        with a warning instead of silently hiding the fallback.
        """
        candidates: list[Path] = []
        for filename in cls.CALIBRATION_ARTIFACT_FILES:
            candidates.append(root / filename)
            candidates.append(root / "data" / filename)

        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"calibration artifact {path.relative_to(root)} invalid json: {exc}")
                continue
            except OSError as exc:
                errors.append(f"calibration artifact {path.relative_to(root)} unreadable: {exc}")
                continue
            if not isinstance(payload, dict):
                errors.append(f"calibration artifact {path.relative_to(root)} must decode to an object")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = ConfigValidator.validate_all(skip_env=True)
    if ok:
        print("Validation successful")
    else:
        print("Validation FAILED")
