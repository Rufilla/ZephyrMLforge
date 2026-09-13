# _____________________________________________________________________________
#
# @file visualizer.py
# @brief Model visualization utilities
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Model visualization utilities.

Provides functionality to visualize TFLite models using various backends:
- Netron (interactive visualization)
- Text-based summary
- Architecture diagram
"""

from __future__ import annotations

import json
import logging
import subprocess
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)


def _dtype_name(detail: dict) -> str:
    """Readable name for a tensor's dtype, which is a type object or a string."""
    dtype = detail["dtype"]
    return dtype.__name__ if hasattr(dtype, "__name__") else str(dtype)


def visualize_tflite(
    model_path: Path,
    method: str = "netron",
    output_path: Path | None = None,
    open_browser: bool = True,
) -> str | None:
    """
    Visualize a TFLite model.

    Args:
        model_path: Path to .tflite file
        method: Visualization method ('netron', 'summary', 'json')
        output_path: Optional path for output file
        open_browser: Whether to open browser for netron

    Returns:
        Visualization output (string for summary/json, None for netron)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if method == "netron":
        return _visualize_netron(model_path, open_browser)
    elif method == "summary":
        return _generate_summary(model_path, output_path)
    elif method == "json":
        return _generate_json(model_path, output_path)
    else:
        raise ValueError(f"Unknown visualization method: {method}")


def _visualize_netron(model_path: Path, open_browser: bool = True) -> str | None:
    """Open model in Netron viewer."""
    try:
        import netron
        logger.info(f"Opening {model_path} in Netron...")
        if open_browser:
            netron.start(str(model_path))
        else:
            # Return URL without opening browser
            return f"netron://{model_path}"
    except ImportError:
        # Try CLI netron
        logger.info("Netron Python package not installed, trying CLI...")
        try:
            subprocess.run(
                ["netron", str(model_path)],
                check=True,
            )
        except FileNotFoundError:
            # Provide instructions
            logger.warning(
                "Netron not installed. Install with: pip install netron\n"
                "Or open the model at: https://netron.app/"
            )
            if open_browser:
                webbrowser.open("https://netron.app/")
            print(f"\nModel path: {model_path.absolute()}")
            print("Upload this file to https://netron.app/ to visualize")
            return None


def _generate_summary(model_path: Path, output_path: Path | None = None) -> str:
    """Generate text summary of model architecture."""
    try:
        import tensorflow as tf
    except ImportError:
        return "TensorFlow not installed, cannot generate summary"

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get tensor details for all tensors
    tensor_details = interpreter.get_tensor_details()

    lines = []
    lines.append("=" * 60)
    lines.append(f"TFLite Model Summary: {model_path.name}")
    lines.append("=" * 60)
    lines.append("")

    # File info
    model_size = model_path.stat().st_size
    lines.append(f"File size: {model_size:,} bytes ({model_size / 1024:.1f} KB)")
    lines.append("")

    # Input tensors
    lines.append("INPUT TENSORS:")
    lines.append("-" * 40)
    for i, detail in enumerate(input_details):
        dtype = _dtype_name(detail)
        lines.append(f"  [{i}] {detail['name']}")
        lines.append(f"      Shape: {detail['shape'].tolist()}")
        lines.append(f"      Type:  {dtype}")
        if detail.get('quantization') and detail['quantization'][0] != 0:
            scale, zp = detail['quantization']
            lines.append(f"      Quant: scale={scale}, zero_point={zp}")
    lines.append("")

    # Output tensors
    lines.append("OUTPUT TENSORS:")
    lines.append("-" * 40)
    for i, detail in enumerate(output_details):
        dtype = _dtype_name(detail)
        lines.append(f"  [{i}] {detail['name']}")
        lines.append(f"      Shape: {detail['shape'].tolist()}")
        lines.append(f"      Type:  {dtype}")
        if detail.get('quantization') and detail['quantization'][0] != 0:
            scale, zp = detail['quantization']
            lines.append(f"      Quant: scale={scale}, zero_point={zp}")
    lines.append("")

    # All tensors summary
    lines.append("ALL TENSORS:")
    lines.append("-" * 40)
    total_params = 0
    for detail in tensor_details:
        shape = detail['shape'].tolist()
        if len(shape) > 0:
            params = 1
            for dim in shape:
                params *= dim
            total_params += params
        dtype = _dtype_name(detail)
        lines.append(f"  {detail['name']}: {shape} ({dtype})")
    lines.append("")
    lines.append(f"Total tensor elements: {total_params:,}")

    # Architecture visualization (simple)
    lines.append("")
    lines.append("ARCHITECTURE:")
    lines.append("-" * 40)
    lines.append(_generate_ascii_diagram(tensor_details, input_details, output_details))

    lines.append("")
    lines.append("=" * 60)

    summary = "\n".join(lines)

    if output_path:
        output_path.write_text(summary)
        logger.info(f"Summary saved to {output_path}")

    return summary


def _generate_ascii_diagram(
    tensor_details: list,
    input_details: list,
    output_details: list,
) -> str:
    """Generate simple ASCII architecture diagram."""
    lines = []

    # Find layer tensors (those with weights)
    input_names = {d['name'] for d in input_details}
    output_names = {d['name'] for d in output_details}

    # Simple diagram
    lines.append("  ┌─────────────────────┐")
    for detail in input_details:
        shape = detail['shape'].tolist()
        lines.append(f"  │  Input: {shape}      │")
    lines.append("  └──────────┬──────────┘")
    lines.append("             │")

    # Find intermediate layers (simplified)
    for detail in tensor_details:
        name = detail['name']
        if name not in input_names and name not in output_names:
            if 'dense' in name.lower() or 'conv' in name.lower() or 'relu' in name.lower():
                shape = detail['shape'].tolist()
                layer_type = "Dense" if "dense" in name.lower() else "Layer"
                lines.append("             ▼")
                lines.append("  ┌─────────────────────┐")
                lines.append(f"  │ {layer_type}: {shape}".ljust(22) + "│")
                lines.append("  └──────────┬──────────┘")

    lines.append("             │")
    lines.append("             ▼")
    lines.append("  ┌─────────────────────┐")
    for detail in output_details:
        shape = detail['shape'].tolist()
        lines.append(f"  │  Output: {shape}     │")
    lines.append("  └─────────────────────┘")

    return "\n".join(lines)


def _generate_json(model_path: Path, output_path: Path | None = None) -> str:
    """Generate JSON representation of model structure."""
    try:
        import tensorflow as tf
    except ImportError:
        return json.dumps({"error": "TensorFlow not installed"})

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    def serialize_detail(detail: dict) -> dict:
        """Convert numpy types to JSON-serializable types."""
        result = {}
        for key, value in detail.items():
            if hasattr(value, 'tolist'):
                result[key] = value.tolist()
            elif hasattr(value, '__name__'):
                result[key] = value.__name__
            else:
                try:
                    json.dumps(value)
                    result[key] = value
                except (TypeError, ValueError):
                    result[key] = str(value)
        return result

    model_info = {
        "file": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "inputs": [serialize_detail(d) for d in interpreter.get_input_details()],
        "outputs": [serialize_detail(d) for d in interpreter.get_output_details()],
        "tensors": [serialize_detail(d) for d in interpreter.get_tensor_details()],
    }

    json_str = json.dumps(model_info, indent=2)

    if output_path:
        output_path.write_text(json_str)
        logger.info(f"JSON saved to {output_path}")

    return json_str


def print_model_summary(model_path: Path) -> None:
    """Print model summary to console."""
    summary = _generate_summary(model_path)
    print(summary)
