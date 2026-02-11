"""
Model Comparison Manager - MVP Backend
Side-by-side comparison of two loaded models.

Implements core comparison functionality:
- Load two models independently (A and B)
- Generate text from both with same prompt
- Calculate and return comparison metrics
- Memory management and cleanup
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any, TYPE_CHECKING

# Lazy imports for torch/transformers to prevent DLL failures on Windows
# These are imported when actually needed, not at module load time
if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_torch():
    """Lazy import torch."""
    import torch
    return torch


def _get_transformers():
    """Lazy import transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer


class ComparisonManager:
    """Manages side-by-side comparison of two loaded models.

    MVP Features:
    - Load/unload models A and B independently
    - Parallel text generation from both models
    - Basic metrics: token count, generation time, throughput
    - Thread-safe operations
    """

    def __init__(self, logger: logging.Logger):
        """Initialize the comparison manager.

        Args:
            logger: Logger instance for tracking operations
        """
        self.logger = logger

        # Model A state
        self.model_a: Optional[AutoModelForCausalLM] = None
        self.tokenizer_a: Optional[AutoTokenizer] = None
        self.model_a_path: Optional[str] = None

        # Model B state
        self.model_b: Optional[AutoModelForCausalLM] = None
        self.tokenizer_b: Optional[AutoTokenizer] = None
        self.model_b_path: Optional[str] = None

        # Thread management
        self._generation_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="comparison")

        self.logger.info("ComparisonManager initialized")

    def load_model_a(self, model_path: str) -> bool:
        """Load first model for comparison.

        Args:
            model_path: Path to model directory

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading Model A from: {model_path}")

            # Lazy import torch and transformers
            torch = _get_torch()
            AutoModelForCausalLM, AutoTokenizer = _get_transformers()

            abs_path = os.path.abspath(model_path)
            if not os.path.exists(abs_path):
                self.logger.error(f"Model path not found: {abs_path}")
                return False

            # Load tokenizer
            self.tokenizer_a = AutoTokenizer.from_pretrained(abs_path)
            if self.tokenizer_a.pad_token is None:
                self.tokenizer_a.pad_token = self.tokenizer_a.eos_token

            # Load model with memory optimization
            self.model_a = AutoModelForCausalLM.from_pretrained(
                abs_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )

            self.model_a_path = abs_path
            self.logger.info(f"Model A loaded successfully: {abs_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load Model A: {e}", exc_info=True)
            self.model_a = None
            self.tokenizer_a = None
            self.model_a_path = None
            return False

    def load_model_b(self, model_path: str) -> bool:
        """Load second model for comparison.

        Args:
            model_path: Path to model directory

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading Model B from: {model_path}")

            # Lazy import torch and transformers
            torch = _get_torch()
            AutoModelForCausalLM, AutoTokenizer = _get_transformers()

            abs_path = os.path.abspath(model_path)
            if not os.path.exists(abs_path):
                self.logger.error(f"Model path not found: {abs_path}")
                return False

            # Load tokenizer
            self.tokenizer_b = AutoTokenizer.from_pretrained(abs_path)
            if self.tokenizer_b.pad_token is None:
                self.tokenizer_b.pad_token = self.tokenizer_b.eos_token

            # Load model with memory optimization
            self.model_b = AutoModelForCausalLM.from_pretrained(
                abs_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )

            self.model_b_path = abs_path
            self.logger.info(f"Model B loaded successfully: {abs_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load Model B: {e}", exc_info=True)
            self.model_b = None
            self.tokenizer_b = None
            self.model_b_path = None
            return False

    def _generate_single(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from a single model.

        Args:
            model: The model to generate from
            tokenizer: The tokenizer for the model
            prompt: Input prompt text
            **kwargs: Generation parameters (max_tokens, temperature, etc.)

        Returns:
            Dictionary with output text and metrics
        """
        try:
            torch = _get_torch()
            start_time = time.time()

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_length = inputs['input_ids'].shape[1]

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', 100),
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.9),
                    top_k=kwargs.get('top_k', 50),
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode output
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt from response
            response = full_response[len(prompt):].strip()

            # Calculate metrics
            time_taken = time.time() - start_time
            token_count = outputs[0].shape[0] - input_length
            tokens_per_sec = token_count / time_taken if time_taken > 0 else 0

            return {
                'output': response,
                'tokens': int(token_count),
                'time_taken': float(time_taken),
                'tokens_per_sec': float(tokens_per_sec),
                'status': 'success'
            }

        except Exception as e:
            self.logger.error(f"Generation error: {e}", exc_info=True)
            return {
                'output': '',
                'tokens': 0,
                'time_taken': 0.0,
                'tokens_per_sec': 0.0,
                'status': 'error',
                'error': str(e)
            }

    def generate_comparison(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Dict[str, any]:
        """Generate from both models in parallel and return comparison.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Dictionary containing outputs from both models and comparison metrics
        """
        if self.model_a is None or self.model_b is None:
            return {
                'success': False,
                'error': 'Both Model A and Model B must be loaded before comparison'
            }

        if not prompt or not prompt.strip():
            return {
                'success': False,
                'error': 'Prompt cannot be empty'
            }

        try:
            self.logger.info(f"Starting comparison generation with prompt: {prompt[:50]}...")

            # Generation parameters
            gen_kwargs = {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k
            }

            # Submit parallel generation tasks
            future_a = self._executor.submit(
                self._generate_single,
                self.model_a,
                self.tokenizer_a,
                prompt,
                **gen_kwargs
            )

            future_b = self._executor.submit(
                self._generate_single,
                self.model_b,
                self.tokenizer_b,
                prompt,
                **gen_kwargs
            )

            # Wait for both to complete
            result_a = future_a.result()
            result_b = future_b.result()

            # Check for errors
            if result_a['status'] == 'error' or result_b['status'] == 'error':
                return {
                    'success': False,
                    'error': f"Generation failed - A: {result_a.get('error', 'OK')}, B: {result_b.get('error', 'OK')}"
                }

            # Calculate comparison metrics
            metrics = self._calculate_metrics(result_a, result_b)

            self.logger.info("Comparison generation completed successfully")

            return {
                'success': True,
                'model_a': result_a,
                'model_b': result_b,
                'metrics': metrics
            }

        except Exception as e:
            self.logger.error(f"Comparison generation failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': f"Comparison failed: {str(e)}"
            }

    def _calculate_metrics(self, result_a: Dict, result_b: Dict) -> Dict[str, any]:
        """Calculate comparison metrics between two results.

        Args:
            result_a: Generation result from Model A
            result_b: Generation result from Model B

        Returns:
            Dictionary of comparison metrics
        """
        length_diff = result_b['tokens'] - result_a['tokens']
        time_diff = result_b['time_taken'] - result_a['time_taken']

        # Calculate speed difference percentage
        if result_a['tokens_per_sec'] > 0:
            speed_diff_percent = (
                (result_b['tokens_per_sec'] / result_a['tokens_per_sec'] - 1) * 100
            )
        else:
            speed_diff_percent = 0.0

        return {
            'length_diff': int(length_diff),
            'time_diff': float(time_diff),
            'speed_diff_percent': float(speed_diff_percent)
        }

    def get_comparison_status(self) -> Dict[str, any]:
        """Get current status of both models.

        Returns:
            Dictionary with loading status and paths
        """
        return {
            'model_a_loaded': self.model_a is not None,
            'model_a_path': self.model_a_path,
            'model_b_loaded': self.model_b is not None,
            'model_b_path': self.model_b_path
        }

    def unload_model_a(self) -> None:
        """Unload Model A and free memory."""
        if self.model_a is not None:
            self.logger.info("Unloading Model A")
            del self.model_a
            del self.tokenizer_a

            try:
                torch = _get_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass  # torch not available, skip CUDA cleanup

            self.model_a = None
            self.tokenizer_a = None
            self.model_a_path = None
            self.logger.info("Model A unloaded")

    def unload_model_b(self) -> None:
        """Unload Model B and free memory."""
        if self.model_b is not None:
            self.logger.info("Unloading Model B")
            del self.model_b
            del self.tokenizer_b

            try:
                torch = _get_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass  # torch not available, skip CUDA cleanup

            self.model_b = None
            self.tokenizer_b = None
            self.model_b_path = None
            self.logger.info("Model B unloaded")

    def unload_all(self) -> None:
        """Unload both models and free all memory."""
        self.logger.info("Unloading all comparison models")
        self.unload_model_a()
        self.unload_model_b()
        self.logger.info("All comparison models unloaded")

    def cleanup(self) -> None:
        """Cleanup resources and shutdown executor."""
        self.logger.info("Cleaning up ComparisonManager")
        self.unload_all()
        self._executor.shutdown(wait=True)
        self.logger.info("ComparisonManager cleanup complete")
