"""Pricing calculations for Claude models."""

import logging
from typing import Any, Dict, List, Optional

from .pricing_fetcher import LiteLLMPricingFetcher

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (in USD)
# Using base model names without timestamps for better compatibility
MODEL_PRICING = {
    "opus-4": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "opus-3": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "sonnet-4": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "sonnet-3.5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "haiku-3.5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    "haiku-3": {"input": 0.25, "output": 1.25, "cache_write": 0.30, "cache_read": 0.03},
    # Default pricing for unknown models
    "default": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
}


def normalize_model_name(model: str) -> str:
    """Normalize model name to match pricing keys."""
    if not model:
        return "default"

    model_lower = model.lower()

    # Extract key parts
    if "opus-4" in model_lower or "opus 4" in model_lower:
        return "opus-4"
    elif "opus" in model_lower:
        return "opus-3"
    elif "sonnet-4" in model_lower or "sonnet 4" in model_lower:
        return "sonnet-4"
    elif "sonnet" in model_lower:
        return "sonnet-3.5"
    elif "haiku" in model_lower and "3.5" in model_lower:
        return "haiku-3.5"
    elif "haiku" in model_lower:
        return "haiku-3"

    return "default"


class CostCalculator:
    """Calculate costs based on token usage."""

    def __init__(
        self,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        use_litellm: bool = True,
    ):
        self.pricing = MODEL_PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)

        # Try to fetch LiteLLM pricing
        if use_litellm:
            self._fetch_litellm_pricing()

    def _fetch_litellm_pricing(self) -> None:
        """Fetch and integrate LiteLLM pricing data."""
        try:
            fetcher = LiteLLMPricingFetcher()
            litellm_pricing = fetcher.fetch_pricing()

            if not litellm_pricing:
                logger.debug("No LiteLLM pricing data available")
                return

            # Map LiteLLM model names to our normalized names
            model_mappings = {
                "claude-opus-4": ["opus-4"],
                "claude-3-opus": ["opus-3"],
                "claude-3-5-sonnet": ["sonnet-3.5"],
                "claude-3-sonnet": ["sonnet-3.5"],
                "claude-3-haiku": ["haiku-3"],
                "claude-3-5-haiku": ["haiku-3.5"],
                "claude-sonnet-4": ["sonnet-4"],
            }

            # Update pricing for known models
            updated_count = 0
            for litellm_pattern, normalized_names in model_mappings.items():
                # Search for models matching the pattern
                for model_name, pricing_obj in litellm_pricing.items():
                    if litellm_pattern in model_name.lower():
                        costs = pricing_obj.to_million_token_costs()
                        # Update all normalized names
                        for normalized in normalized_names:
                            self.pricing[normalized] = costs
                            updated_count += 1
                            logger.debug(
                                f"Updated {normalized} pricing from {model_name}"
                            )
                        break

            if updated_count > 0:
                logger.info(f"Updated pricing for {updated_count} models from LiteLLM")

        except Exception as e:
            logger.debug(f"Failed to fetch LiteLLM pricing: {e}")

    def calculate_costs(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate costs for all models and sessions."""
        costs: Dict[str, Any] = {
            "by_model": {},
            "by_session": {},
            "by_date": {},
            "total": {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "cache_write_cost": 0.0,
                "cache_read_cost": 0.0,
                "total_cost": 0.0,
            },
        }

        # Calculate costs by model
        for model, usage in stats.get("by_model", {}).items():
            model_costs = self._calculate_model_costs(model, usage)
            costs["by_model"][model] = model_costs

            # Update totals
            for key in [
                "input_cost",
                "output_cost",
                "cache_write_cost",
                "cache_read_cost",
                "total_cost",
            ]:
                costs["total"][key] += model_costs[key]

        # Calculate costs by session
        for session_id, session_data in stats.get("by_session", {}).items():
            # Use model-specific pricing if only one model was used
            models_used = session_data.get("models_used", [])
            if len(models_used) == 1:
                session_cost = self._calculate_model_costs(
                    list(models_used)[0], session_data
                )
            else:
                # For mixed models, calculate weighted average based on the most expensive model used
                session_cost = self._calculate_weighted_costs(session_data, models_used)
            costs["by_session"][session_id] = session_cost

        # Calculate costs by date
        for date, date_data in stats.get("by_date", {}).items():
            # For daily costs, sum up actual costs from all models used that day
            date_cost = self._calculate_weighted_costs(date_data, [])
            costs["by_date"][date] = date_cost

        return costs

    def _calculate_model_costs(
        self, model: str, usage: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate costs for a specific model."""
        normalized_model = normalize_model_name(model)
        model_pricing = self.pricing.get(normalized_model, self.pricing["default"])

        input_cost = (usage.get("input_tokens", 0) / 1_000_000) * model_pricing["input"]
        output_cost = (usage.get("output_tokens", 0) / 1_000_000) * model_pricing[
            "output"
        ]
        cache_write_cost = (
            usage.get("cache_creation_input_tokens", 0) / 1_000_000
        ) * model_pricing["cache_write"]
        cache_read_cost = (
            usage.get("cache_read_input_tokens", 0) / 1_000_000
        ) * model_pricing["cache_read"]

        total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cache_write_cost": cache_write_cost,
            "cache_read_cost": cache_read_cost,
            "total_cost": total_cost,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        }

    def _calculate_weighted_costs(
        self, usage: Dict[str, Any], models_used: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate costs using the most expensive model pricing or weighted average."""
        # If models are specified, use the most expensive one
        if models_used:
            # Find the most expensive model based on output pricing
            normalized_models = [normalize_model_name(m) for m in models_used]
            most_expensive_model = max(
                normalized_models,
                key=lambda m: self.pricing.get(m, self.pricing["default"])["output"],
            )
            model_pricing = self.pricing.get(
                most_expensive_model, self.pricing["default"]
            )
        else:
            # Use Opus pricing as it's the most expensive (conservative estimate)
            model_pricing = self.pricing.get("opus-4", self.pricing["default"])

        input_cost = (usage.get("input_tokens", 0) / 1_000_000) * model_pricing["input"]
        output_cost = (usage.get("output_tokens", 0) / 1_000_000) * model_pricing[
            "output"
        ]
        cache_write_cost = (
            usage.get("cache_creation_input_tokens", 0) / 1_000_000
        ) * model_pricing["cache_write"]
        cache_read_cost = (
            usage.get("cache_read_input_tokens", 0) / 1_000_000
        ) * model_pricing["cache_read"]

        total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cache_write_cost": cache_write_cost,
            "cache_read_cost": cache_read_cost,
            "total_cost": total_cost,
        }
