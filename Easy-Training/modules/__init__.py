"""
Module Loader - Auto-discovers and loads training modules.

To add a new module:
1. Create a .py file in this directory
2. Define MODULE_INFO dict with name, description, config_schema
3. Optionally define register_routes(app, logger) and/or training_hook(...)

The loader will automatically find and register your module.
"""

import os
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional

# Registry of loaded modules.
_modules: Dict[str, dict] = {}


def load_modules(app=None, logger=None) -> Dict[str, dict]:
    """
    Discover and load all modules in this directory.

    Returns dict of {module_name: module_info}
    """
    global _modules

    modules_dir = Path(__file__).parent

    for file in modules_dir.glob("*.py"):
        if file.name.startswith("_"):
            continue

        module_name = file.stem

        try:
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check for MODULE_INFO
            if hasattr(module, "MODULE_INFO"):
                info = module.MODULE_INFO.copy()
                info["_module"] = module
                info["_file"] = str(file)

                # Register routes if available
                if app and hasattr(module, "register_routes"):
                    try:
                        module.register_routes(app, logger)
                        info["_routes_registered"] = True
                        if logger:
                            logger.info(f"Module '{info.get('name', module_name)}' routes registered")
                    except Exception as e:
                        info["_routes_error"] = str(e)
                        if logger:
                            logger.warning(f"Module '{module_name}' route registration failed: {e}")

                _modules[module_name] = info

                if logger:
                    logger.info(f"Loaded module: {info.get('name', module_name)}")

        except Exception as e:
            if logger:
                logger.warning(f"Failed to load module {module_name}: {e}")

    return _modules


def get_module(name: str) -> Optional[dict]:
    """Get a loaded module by name."""
    return _modules.get(name)


def get_all_modules() -> Dict[str, dict]:
    """Get all loaded modules."""
    return _modules.copy()


def get_training_hooks() -> List[Callable]:
    """Get all training hooks from loaded modules."""
    hooks = []
    for name, info in _modules.items():
        module = info.get("_module")
        if module and hasattr(module, "training_hook"):
            hooks.append(module.training_hook)
    return hooks


def get_module_configs() -> Dict[str, dict]:
    """Get config schemas for all modules (for UI generation)."""
    configs = {}
    for name, info in _modules.items():
        if "config_schema" in info:
            configs[name] = {
                "name": info.get("name", name),
                "description": info.get("description", ""),
                "schema": info["config_schema"]
            }
    return configs
