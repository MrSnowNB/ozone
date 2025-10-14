#!/usr/bin/env python3
"""
O3 Hardware Monitor - Advanced Resource Tracking for AI-First Optimization
Phase 2: Real-time peak detection, AMD GPU support, safety thresholds
"""

import subprocess
import threading
import time
import psutil
import os
from typing import Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import yaml
import datetime


@dataclass
class HardwareMetrics:
    """Real-time hardware metrics snapshot"""
    timestamp: float
    vram_used_mb: Optional[int]
    vram_total_mb: Optional[int]
    ram_used_mb: int
    ram_total_gb: float
    gpu_temp_c: Optional[float]
    gpu_utilization_percent: Optional[float]
    cpu_percent: float
    cpu_temp_c: Optional[float]

    @property
    def vram_utilization_percent(self) -> Optional[float]:
        """Calculate VRAM utilization percentage"""
        if self.vram_used_mb and self.vram_total_mb:
            return (self.vram_used_mb / self.vram_total_mb) * 100
        return None

    @property
    def ram_utilization_percent(self) -> float:
        """Calculate RAM utilization percentage"""
        return (self.ram_used_mb / (self.ram_total_gb * 1024)) * 100


class RealTimeMonitor:
    """Real-time hardware monitoring with peak detection"""

    def __init__(self, sampling_interval_ms: int = 100):
        self.sampling_interval = sampling_interval_ms / 1000.0  # Convert to seconds
        self.monitoring = False
        self.peak_metrics = None
        self.current_metrics = None
        self.samples = []
        self.callbacks: Dict[str, Callable] = {}

        # Hardware detection
        self.hardware_monitor = HardwareMonitor()

    def start_monitoring(self) -> None:
        """Start real-time monitoring thread"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop_monitoring(self) -> HardwareMetrics:
        """Stop monitoring and return peak metrics"""
        self.monitoring = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        return self.peak_metrics

    def add_threshold_callback(self, name: str, callback: Callable) -> None:
        """Add callback to trigger when thresholds are exceeded"""
        self.callbacks[name] = callback

    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.hardware_monitor.get_current_metrics()

                # Update current metrics
                self.current_metrics = metrics

                # Track peak values
                self._update_peaks(metrics)

                # Store samples for analysis
                self.samples.append(metrics)
                if len(self.samples) > 100:  # Keep last 100 samples (10 seconds at 100ms)
                    self.samples.pop(0)

                # Check thresholds and trigger callbacks
                self._check_thresholds(metrics)

                time.sleep(self.sampling_interval)

            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(self.sampling_interval)

    def _update_peaks(self, metrics: HardwareMetrics) -> None:
        """Update peak metrics tracking"""
        if not self.peak_metrics:
            self.peak_metrics = metrics
            return

        # Update peaks
        if metrics.vram_used_mb and (not self.peak_metrics.vram_used_mb or
                                   metrics.vram_used_mb > self.peak_metrics.vram_used_mb):
            self.peak_metrics.vram_used_mb = metrics.vram_used_mb

        if metrics.gpu_temp_c and (not self.peak_metrics.gpu_temp_c or
                                  metrics.gpu_temp_c > self.peak_metrics.gpu_temp_c):
            self.peak_metrics.gpu_temp_c = metrics.gpu_temp_c

        if metrics.gpu_utilization_percent and (not self.peak_metrics.gpu_utilization_percent or
                                              metrics.gpu_utilization_percent > self.peak_metrics.gpu_utilization_percent):
            self.peak_metrics.gpu_utilization_percent = metrics.gpu_utilization_percent

        if metrics.cpu_percent and (not self.peak_metrics.cpu_percent or
                                   metrics.cpu_percent > self.peak_metrics.cpu_percent):
            self.peak_metrics.cpu_percent = metrics.cpu_percent

    def _check_thresholds(self, metrics: HardwareMetrics) -> None:
        """Check safety thresholds and trigger callbacks"""
        ai_config = self._load_thresholds()

        if not ai_config:
            return

        thresholds = ai_config.get("safety_thresholds", {})

        # VRAM utilization threshold
        vram_percent = metrics.vram_utilization_percent
        if vram_percent and vram_percent > thresholds.get("max_vram_percent", 90):
            if "vram_threshold" in self.callbacks:
                self.callbacks["vram_threshold"](f"VRAM utilization {vram_percent:.1f}% exceeds {thresholds['max_vram_percent']}%")

        # RAM utilization threshold
        ram_percent = metrics.ram_utilization_percent
        if ram_percent > thresholds.get("max_ram_percent", 85):
            if "ram_threshold" in self.callbacks:
                self.callbacks["ram_threshold"](f"RAM utilization {ram_percent:.1f}% exceeds {thresholds['max_ram_percent']}%")

        # Temperature thresholds
        if metrics.gpu_temp_c and metrics.gpu_temp_c > thresholds.get("max_temperature_c", 85):
            if "temp_threshold" in self.callbacks:
                self.callbacks["temp_threshold"](f"GPU temperature {metrics.gpu_temp_c:.1f}°C exceeds {thresholds['max_temperature_c']}°C")

        # CPU utilization threshold
        if metrics.cpu_percent > thresholds.get("max_cpu_percent", 90):
            if "cpu_threshold" in self.callbacks:
                self.callbacks["cpu_threshold"](f"CPU utilization {metrics.cpu_percent:.1f}% exceeds {thresholds['max_cpu_percent']}%")

    def _load_thresholds(self) -> Dict:
        """Load safety thresholds from AI config"""
        try:
            config_path = Path("o3_ai_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception:
            pass
        return {}


class HardwareMonitor:
    """Enhanced hardware monitoring with AMD GPU support and safety thresholds"""

    def __init__(self):
        self.gpu_type = self._detect_gpu()
        self.vram_total_mb = self._get_vram_total()
        self.real_time_monitor = None

    def _detect_gpu(self) -> str:
        """Detect GPU type (AMD/NVIDIA/None) with enhanced detection"""
        try:
            # Try NVIDIA first
            result = subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
            return "nvidia"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            # Try AMD ROCm
            result = subprocess.run(["rocm-smi"], capture_output=True, check=True, timeout=5)
            return "amd"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return "none"

    def _get_vram_total(self) -> Optional[int]:
        """Get total VRAM in MB"""
        if self.gpu_type == "nvidia":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=5
                )
                return int(result.stdout.strip())
            except Exception:
                return None

        elif self.gpu_type == "amd":
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True, text=True, check=True, timeout=5
                )
                # Parse AMD output for total VRAM
                for line in result.stdout.split('\n'):
                    if 'Total VRAM' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            vram_str = parts[2]
                            if vram_str.endswith('MB'):
                                return int(vram_str[:-2])
                            elif vram_str.endswith('GB'):
                                return int(float(vram_str[:-2]) * 1024)
                return None
            except Exception:
                return None

        return None

    def get_vram_usage(self) -> Optional[int]:
        """Get current VRAM usage in MB with improved AMD support"""
        if self.gpu_type == "nvidia":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=2
                )
                return int(result.stdout.strip())
            except Exception:
                return None

        elif self.gpu_type == "amd":
            try:
                # Use more reliable AMD VRAM query
                result = subprocess.run(
                    ["rocm-smi", "--showmemuse", "--csv"],
                    capture_output=True, text=True, check=True, timeout=2
                )

                # Parse CSV format: Device,VRAM Total Memory (B),VRAM Total Used (B)
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Skip header, use first device
                    data = lines[1].split(',')
                    if len(data) >= 3:
                        vram_used_bytes = int(data[2])  # VRAM Total Used (B)
                        return vram_used_bytes // (1024 * 1024)  # Convert to MB

                return None
            except Exception:
                return None

        return None

    def get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature"""
        if self.gpu_type == "nvidia":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=2
                )
                return float(result.stdout.strip())
            except Exception:
                return None

        elif self.gpu_type == "amd":
            try:
                # AMD temperature monitoring
                result = subprocess.run(
                    ["rocm-smi", "--showtemp", "--csv"],
                    capture_output=True, text=True, check=True, timeout=2
                )

                # Parse AMD temperature CSV
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Use first device
                    data = lines[1].split(',')
                    if len(data) >= 2:
                        return float(data[1])  # Temperature column

                return None
            except Exception:
                return None

        return None

    def get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        if self.gpu_type == "nvidia":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=2
                )
                return float(result.stdout.strip())
            except Exception:
                return None

        elif self.gpu_type == "amd":
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showuse", "--csv"],
                    capture_output=True, text=True, check=True, timeout=2
                )

                # Parse AMD utilization CSV
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Use first device
                    data = lines[1].split(',')
                    if len(data) >= 2:
                        return float(data[1])  # GPU Usage column

                return None
            except Exception:
                return None

        return None

    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature using various methods"""
        # Try psutil first (most reliable)
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps and temps['coretemp']:
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps and temps['cpu_thermal']:
                return temps['cpu_thermal'][0].current
        except Exception:
            pass

        # Fallback to system commands
        try:
            if os.name == 'nt':  # Windows
                # Windows CPU temperature is harder to get reliably
                return None
            else:  # Linux/Unix
                result = subprocess.run(
                    ["sensors"],
                    capture_output=True, text=True, timeout=2
                )
                # Parse lm-sensors output for CPU temp
                for line in result.stdout.split('\n'):
                    if 'Tctl:' in line or 'CPU Temperature:' in line:
                        temp_str = line.split(':')[1].strip().split()[0]
                        try:
                            return float(temp_str[:-3])  # Remove °C
                        except ValueError:
                            continue
        except Exception:
            pass

        return None

    def get_ram_usage(self) -> int:
        """Get current RAM usage in MB"""
        return int(psutil.virtual_memory().used / 1024 / 1024)

    def get_current_metrics(self) -> HardwareMetrics:
        """Get comprehensive current hardware metrics"""
        return HardwareMetrics(
            timestamp=time.time(),
            vram_used_mb=self.get_vram_usage(),
            vram_total_mb=self.vram_total_mb,
            ram_used_mb=self.get_ram_usage(),
            ram_total_gb=round(psutil.virtual_memory().total / (1024**3), 2),
            gpu_temp_c=self.get_gpu_temperature(),
            gpu_utilization_percent=self.get_gpu_utilization(),
            cpu_percent=psutil.cpu_percent(),
            cpu_temp_c=self.get_cpu_temperature()
        )

    def create_real_time_monitor(self, sampling_interval_ms: int = 100) -> RealTimeMonitor:
        """Create and return a real-time monitor instance"""
        self.real_time_monitor = RealTimeMonitor(sampling_interval_ms)
        return self.real_time_monitor

    def check_safety_thresholds(self) -> Tuple[bool, Optional[str]]:
        """Check if current hardware usage exceeds safety thresholds"""
        try:
            config_path = Path("o3_ai_config.yaml")
            if not config_path.exists():
                return True, None  # No config = no restrictions

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            thresholds = config.get("safety_thresholds", {})
            metrics = self.get_current_metrics()

            # Check VRAM
            vram_percent = metrics.vram_utilization_percent
            if vram_percent and vram_percent > thresholds.get("max_vram_percent", 90):
                return False, f"VRAM utilization {vram_percent:.1f}% exceeds threshold {thresholds['max_vram_percent']}%"

            # Check RAM
            ram_percent = metrics.ram_utilization_percent
            if ram_percent > thresholds.get("max_ram_percent", 85):
                return False, f"RAM utilization {ram_percent:.1f}% exceeds threshold {thresholds['max_ram_percent']}%"

            # Check temperature
            if metrics.gpu_temp_c and metrics.gpu_temp_c > thresholds.get("max_temperature_c", 85):
                return False, f"GPU temperature {metrics.gpu_temp_c:.1f}°C exceeds threshold {thresholds['max_temperature_c']}°C"

            # Check CPU
            if metrics.cpu_percent > thresholds.get("max_cpu_percent", 90):
                return False, f"CPU utilization {metrics.cpu_percent:.1f}% exceeds threshold {thresholds['max_cpu_percent']}%"

        except Exception as e:
            # On configuration error, allow testing to continue
            return True, f"Configuration error: {e}"

        return True, None

    def log_hardware_snapshot(self, test_id: str = None) -> Dict:
        """Log current hardware state for debugging/optimization"""
        metrics = self.get_current_metrics()
        snapshot = {
            "test_id": test_id or "unknown",
            "timestamp": metrics.timestamp,
            "gpu_type": self.gpu_type,
            "hardware_metrics": {
                "vram_used_mb": metrics.vram_used_mb,
                "vram_total_mb": metrics.vram_total_mb,
                "vram_utilization_percent": metrics.vram_utilization_percent,
                "ram_used_mb": metrics.ram_used_mb,
                "ram_total_gb": metrics.ram_total_gb,
                "ram_utilization_percent": metrics.ram_utilization_percent,
                "gpu_temp_c": metrics.gpu_temp_c,
                "gpu_utilization_percent": metrics.gpu_utilization_percent,
                "cpu_percent": metrics.cpu_percent,
                "cpu_temp_c": metrics.cpu_temp_c
            }
        }

        # Save to environment log if results directory exists
        try:
            env_dir = Path("o3_results/env")
            env_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = env_dir / f"hardware_snapshot_{timestamp}.json"

            with open(log_file, 'w') as f:
                json.dump(snapshot, f, indent=2)

        except Exception:
            pass  # Don't fail if logging fails

        return snapshot


# Convenience function for quick hardware checks
def quick_hardware_check() -> Dict:
    """Quick hardware status check for system validation"""
    monitor = HardwareMonitor()
    metrics = monitor.get_current_metrics()
    safe, reason = monitor.check_safety_thresholds()

    return {
        "hardware_safe": safe,
        "safety_message": reason,
        "gpu_type": monitor.gpu_type,
        "metrics": {
            "vram_utilization": f"{metrics.vram_utilization_percent:.1f}%" if metrics.vram_utilization_percent else "N/A",
            "ram_utilization": f"{metrics.ram_utilization_percent:.1f}%",
            "gpu_temp": f"{metrics.gpu_temp_c:.1f}°C" if metrics.gpu_temp_c else "N/A",
            "cpu_percent": f"{metrics.cpu_percent:.1f}%"
        }
    }


if __name__ == "__main__":
    # Quick hardware status check when run directly
    try:
        print("🖥️  O3 Hardware Monitor - System Check")
        print("=" * 50)

        status = quick_hardware_check()

        print(f"GPU Type: {status['gpu_type']}")
        print(f"Hardware Safe: {'✅' if status['hardware_safe'] else '❌'}")

        if status['safety_message']:
            print(f"Safety Status: {status['safety_message']}")

        metrics = status['metrics']
        print("\nCurrent Metrics:")
        print(f"  VRAM Utilization: {metrics['vram_utilization']}")
        print(f"  RAM Utilization: {metrics['ram_utilization']}")
        print(f"  GPU Temperature: {metrics['gpu_temp']}")
        print(f"  CPU Utilization: {metrics['cpu_percent']}")

        print("\n✅ Hardware monitor ready for real-time peak detection")
    except Exception as e:
        print(f"⚠️  Hardware check error: {e}")
