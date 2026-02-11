"""
Module Template - Copy this to create a new training module.

File: modules/your_module_name.py
"""

# =============================================================================
# MODULE INFO (Required)
# =============================================================================
MODULE_INFO = {
    "name": "Your Module Name",
    "description": "What this module does",
    "version": "1.0.0",
    "author": "Your Name",

    # Config schema - defines what settings appear in the UI
    "config_schema": {
        "enabled": {
            "type": "bool",
            "default": False,
            "description": "Enable this module"
        },
        "strength": {
            "type": "float",
            "default": 0.1,
            "min": 0.0,
            "max": 1.0,
            "description": "How strongly to apply the effect"
        },
        "mode": {
            "type": "select",
            "options": ["mild", "medium", "aggressive"],
            "default": "medium",
            "description": "Operating mode"
        }
    }
}


# =============================================================================
# ROUTES (Optional) - Add API endpoints
# =============================================================================
def register_routes(app, logger=None):
    """Register Flask routes for this module."""

    @app.route("/api/your_module/status")
    def your_module_status():
        return {"status": "ok", "module": MODULE_INFO["name"]}

    # Add more routes as needed...

    if logger:
        logger.info(f"{MODULE_INFO['name']} routes registered")


# =============================================================================
# TRAINING HOOK (Optional) - Called during training
# =============================================================================
def training_hook(
    model,           # The model being trained
    batch,           # Current batch (input_ids, attention_mask, labels)
    step: int,       # Current training step
    config: dict,    # Module config (from config_schema)
    **kwargs         # Additional context (optimizer, scheduler, etc.)
):
    """
    Called every training step. Return modified batch or None.

    Examples:
    - Modify gradients after backward
    - Add regularization
    - Log custom metrics
    - Apply quantization noise
    """
    if not config.get("enabled", False):
        return None

    # Your logic here...
    # strength = config.get("strength", 0.1)

    return None  # or return modified batch


# =============================================================================
# UTILITY FUNCTIONS (Optional)
# =============================================================================
def some_helper_function():
    """Add any helper functions your module needs."""
    pass
