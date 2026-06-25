import json
import re
from pathlib import Path
from typing import Any

from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RuleValidator:
    """
    Validates rows of payload data against business guardrails configured in business_rules/rules.json.
    """

    def __init__(self, rules_path: str | None = None):
        if rules_path is None:
            # Resolve relative to the project root (app/core/feature2_mutation/ -> ../.. -> project root)
            rules_path = str(
                Path(__file__).resolve().parent.parent.parent.parent
                / "business_rules"
                / "rules.json"
            )
        self.rules_path = rules_path
        self.rules = {}
        self.load_rules()

    def load_rules(self):
        try:
            with open(self.rules_path) as f:
                self.rules = json.load(f)
            logger.info("Successfully loaded business validation rules")
        except Exception as e:
            logger.error(f"Failed to load business rules from {self.rules_path}: {e}")

    @track(name="rule_validator_validate")
    def validate_rows(
        self, table_name: str, rows: list[dict[str, Any]]
    ) -> tuple[bool, list[str]]:
        """
        Validate all rows. Returns (is_valid, list_of_error_messages).
        """
        errors = []
        max_rows = self.rules.get("max_bulk_rows", 1000)

        if len(rows) > max_rows:
            return False, [
                f"Payload rows ({len(rows)}) exceeds maximum allowed bulk operations limit ({max_rows})"
            ]

        table_rules = self.rules.get("field_validation_rules", {}).get(table_name, {})
        if not table_rules:
            # If no rules exist, the rows are valid
            return True, []

        for idx, row in enumerate(rows, 1):
            for field, rule in table_rules.items():
                val = row.get(field)
                if val is None:
                    continue  # Optional checks, handle null safety

                rule_type = rule.get("type")
                message = rule.get(
                    "message", f"Field '{field}' failed validation rule."
                )

                if rule_type == "numeric":
                    try:
                        num_val = float(val)
                        if "min" in rule and num_val < rule["min"]:
                            errors.append(f"Row {idx}: {message} (Value: {val})")
                        if "max" in rule and num_val > rule["max"]:
                            errors.append(f"Row {idx}: {message} (Value: {val})")
                    except ValueError:
                        errors.append(
                            f"Row {idx}: Field '{field}' must be a valid numeric type."
                        )

                elif rule_type == "integer":
                    try:
                        int_val = int(val)
                        if "min" in rule and int_val < rule["min"]:
                            errors.append(f"Row {idx}: {message} (Value: {val})")
                        if "max" in rule and int_val > rule["max"]:
                            errors.append(f"Row {idx}: {message} (Value: {val})")
                    except ValueError:
                        errors.append(
                            f"Row {idx}: Field '{field}' must be a valid integer type."
                        )

                elif rule_type == "regex":
                    pattern = rule.get("pattern")
                    if pattern and not re.match(pattern, str(val)):
                        errors.append(f"Row {idx}: {message} (Value: {val})")

                elif rule_type == "enum":
                    allowed = rule.get("allowed", [])
                    if val not in allowed:
                        errors.append(f"Row {idx}: {message} (Value: {val})")

        is_valid = len(errors) == 0
        return is_valid, errors
