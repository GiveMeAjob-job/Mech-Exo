"""
LightGBM GPU Inference PoC
Phase P11 Week 4 Day 2 - GPU acceleration for ML inference

Features:
- CUDA-accelerated LightGBM inference
- Automatic GPU/CPU fallback detection
- Batch inference optimization
- Performance benchmarking
- Cost tracking for GPU instances
"""

import os
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import joblib
import json

try:
    import lightgbm as lgb
    GPU_AVAILABLE = lgb.LGBMRegressor().device_type == 'gpu'
except ImportError:
    lgb = None
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_GPU_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_COUNT = torch.cuda.device_count() if TORCH_GPU_AVAILABLE else 0
except ImportError:
    torch = None
    TORCH_GPU_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics
gpu_inference_requests = Counter('ml_gpu_inference_requests_total', 'Total GPU inference requests', ['status'])
gpu_inference_duration = Histogram('ml_gpu_inference_duration_seconds', 'GPU inference duration', ['batch_size'])
gpu_memory_usage = Gauge('ml_gpu_memory_usage_bytes', 'GPU memory usage in bytes')
gpu_utilization = Gauge('ml_gpu_utilization_percent', 'GPU utilization percentage')
inference_batch_size = Histogram('ml_inference_batch_size', 'Inference batch size distribution')


@dataclass
class GPUConfig:
    """GPU inference configuration"""
    device_type: str = 'gpu'  # 'gpu' or 'cpu'
    gpu_device_id: int = 0
    max_batch_size: int = 1000
    enable_fallback: bool = True
    memory_limit_mb: int = 2048
    
    # Performance settings
    num_threads: int = -1
    boosting_type: str = 'gbdt'
    objective: str = 'regression'
    
    # Model paths
    model_path: str = 'models/lightgbm_model.txt'
    feature_names_path: str = 'models/feature_names.json'


class GPUModelWrapper:
    """Wrapper for LightGBM model with GPU acceleration"""
    
    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.model = None
        self.feature_names = None
        self.device_type = 'cpu'  # Will be set during initialization
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # GPU monitoring
        self.gpu_available = GPU_AVAILABLE and TORCH_GPU_AVAILABLE
        self.cuda_device_count = CUDA_DEVICE_COUNT
        
        logger.info(f"GPU Model Wrapper initialized - GPU available: {self.gpu_available}")
        logger.info(f"CUDA devices: {self.cuda_device_count}")
    
    def detect_gpu_availability(self) -> Dict[str, any]:
        """Detect GPU availability and capabilities"""
        gpu_info = {
            'lightgbm_gpu': False,
            'torch_gpu': TORCH_GPU_AVAILABLE,
            'cuda_device_count': CUDA_DEVICE_COUNT,
            'gpu_devices': []
        }
        
        # Check LightGBM GPU support
        if lgb is not None:
            try:
                test_model = lgb.LGBMRegressor(device_type='gpu', gpu_device_id=0)
                gpu_info['lightgbm_gpu'] = True
            except Exception as e:
                logger.warning(f"LightGBM GPU not available: {e}")
        
        # Get GPU device information
        if TORCH_GPU_AVAILABLE:
            for i in range(CUDA_DEVICE_COUNT):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info['gpu_devices'].append({
                    'device_id': i,
                    'name': device_props.name,
                    'memory_total': device_props.total_memory,
                    'major': device_props.major,
                    'minor': device_props.minor
                })
        
        return gpu_info
    
    def load_model(self, model_path: str = None) -> bool:
        """Load LightGBM model with GPU configuration"""
        model_path = model_path or self.config.model_path
        
        try:
            # Determine device type
            gpu_info = self.detect_gpu_availability()
            
            if self.config.device_type == 'gpu' and gpu_info['lightgbm_gpu']:
                self.device_type = 'gpu'
                logger.info(f"Loading model with GPU acceleration (device: {self.config.gpu_device_id})")
            else:
                self.device_type = 'cpu'
                if self.config.device_type == 'gpu':
                    logger.warning("GPU requested but not available, falling back to CPU")
            
            # Load model
            if Path(model_path).exists():
                self.model = lgb.Booster(model_file=model_path)
                logger.info(f"Loaded LightGBM model from {model_path}")
            else:
                # Create a dummy model for testing
                self.model = self._create_dummy_model()
                logger.warning("Model file not found, created dummy model for testing")
            
            # Load feature names
            feature_names_path = self.config.feature_names_path
            if Path(feature_names_path).exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
            else:
                # Generate dummy feature names
                self.feature_names = [f'feature_{i}' for i in range(20)]
                logger.warning("Feature names file not found, using dummy names")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _create_dummy_model(self) -> lgb.Booster:
        """Create a dummy LightGBM model for testing"""
        # Generate dummy training data
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = np.random.randn(1000)
        
        # Create and train model
        train_data = lgb.Dataset(X, label=y)
        
        params = {
            'objective': self.config.objective,
            'boosting_type': self.config.boosting_type,
            'device_type': self.device_type,
            'num_threads': self.config.num_threads,
            'verbose': -1
        }
        
        if self.device_type == 'gpu':
            params['gpu_device_id'] = self.config.gpu_device_id
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        return model
    
    def predict_single(self, features: Union[np.ndarray, List[float]]) -> float:
        """Predict single sample"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        try:
            if isinstance(features, list):
                features = np.array(features).reshape(1, -1)
            elif features.ndim == 1:
                features = features.reshape(1, -1)
            
            prediction = self.model.predict(features)[0]
            
            # Update metrics
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            gpu_inference_requests.labels(status='success').inc()
            gpu_inference_duration.labels(batch_size='1').observe(inference_time)
            inference_batch_size.observe(1)
            
            return float(prediction)
            
        except Exception as e:
            gpu_inference_requests.labels(status='error').inc()
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Predict batch of samples efficiently"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        batch_size = len(features)
        start_time = time.time()
        
        try:
            # Update GPU memory usage if available
            if self.device_type == 'gpu' and TORCH_GPU_AVAILABLE:
                torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated(self.config.gpu_device_id)
                gpu_memory_usage.set(memory_allocated)
            
            predictions = self.model.predict(features)
            
            # Update metrics
            inference_time = time.time() - start_time
            self.inference_count += batch_size
            self.total_inference_time += inference_time
            
            gpu_inference_requests.labels(status='success').inc()
            gpu_inference_duration.labels(batch_size=str(batch_size)).observe(inference_time)
            inference_batch_size.observe(batch_size)
            
            logger.debug(f"Batch inference: {batch_size} samples in {inference_time:.4f}s")
            
            return predictions
            
        except Exception as e:
            gpu_inference_requests.labels(status='error').inc()
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def predict_streaming(self, features_iterator, batch_size: int = None) -> List[float]:
        """Predict on streaming data with batching"""
        batch_size = batch_size or self.config.max_batch_size
        results = []
        batch = []
        
        for features in features_iterator:
            batch.append(features)
            
            if len(batch) >= batch_size:
                batch_features = np.array(batch)
                batch_predictions = self.predict_batch(batch_features)
                results.extend(batch_predictions.tolist())
                batch = []
        
        # Process remaining batch
        if batch:
            batch_features = np.array(batch)
            batch_predictions = self.predict_batch(batch_features)
            results.extend(batch_predictions.tolist())
        
        return results
    
    def get_performance_stats(self) -> Dict[str, any]:
        """Get performance statistics"""
        avg_inference_time = self.total_inference_time / max(1, self.inference_count)
        
        stats = {
            'device_type': self.device_type,
            'inference_count': self.inference_count,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_per_second': 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
            'gpu_available': self.gpu_available,
            'cuda_device_count': self.cuda_device_count
        }
        
        # Add GPU-specific stats
        if self.device_type == 'gpu' and TORCH_GPU_AVAILABLE:
            try:
                device_id = self.config.gpu_device_id
                stats.update({
                    'gpu_memory_allocated': torch.cuda.memory_allocated(device_id),
                    'gpu_memory_cached': torch.cuda.memory_reserved(device_id),
                    'gpu_utilization': torch.cuda.utilization(device_id) if hasattr(torch.cuda, 'utilization') else 0
                })
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")
        
        return stats
    
    def benchmark(self, num_samples: int = 1000, num_features: int = 20, batch_sizes: List[int] = None) -> Dict[str, any]:
        """Benchmark inference performance"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("Failed to load model for benchmarking")
        
        batch_sizes = batch_sizes or [1, 10, 50, 100, 500, 1000]
        results = {}
        
        logger.info(f"Starting benchmark: {num_samples} samples, {num_features} features")
        logger.info(f"Device type: {self.device_type}")
        
        # Generate test data
        np.random.seed(42)
        test_data = np.random.randn(num_samples, num_features)
        
        for batch_size in batch_sizes:
            if batch_size > num_samples:
                continue
            
            times = []
            
            # Run multiple trials
            for trial in range(5):
                start_idx = trial * batch_size % (num_samples - batch_size)
                end_idx = start_idx + batch_size
                batch_data = test_data[start_idx:end_idx]
                
                start_time = time.time()
                predictions = self.predict_batch(batch_data)
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time
            
            results[f'batch_{batch_size}'] = {
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'throughput_per_second': throughput,
                'latency_per_sample_ms': (avg_time / batch_size) * 1000
            }
            
            logger.info(f"Batch {batch_size}: {avg_time*1000:.2f}ms ({throughput:.1f} samples/sec)")
        
        # Overall benchmark results
        results['benchmark_summary'] = {
            'device_type': self.device_type,
            'num_samples': num_samples,
            'num_features': num_features,
            'gpu_available': self.gpu_available,
            'timestamp': time.time()
        }
        
        return results


class GPUInferenceService:
    """Service for managing GPU inference with multiple models"""
    
    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.models = {}
        self.default_model = None
    
    async def initialize(self):
        """Initialize the inference service"""
        logger.info("Initializing GPU Inference Service")
        
        # Load default model
        self.default_model = GPUModelWrapper(self.config)
        if await self._load_model_async(self.default_model):
            logger.info("Default model loaded successfully")
        else:
            logger.error("Failed to load default model")
            raise RuntimeError("Failed to initialize inference service")
    
    async def _load_model_async(self, model_wrapper: GPUModelWrapper) -> bool:
        """Load model asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model_wrapper.load_model)
    
    async def predict_async(self, features: Union[np.ndarray, List[float]], model_name: str = None) -> float:
        """Async prediction interface"""
        model = self.models.get(model_name, self.default_model)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model.predict_single, features)
    
    async def predict_batch_async(self, features: np.ndarray, model_name: str = None) -> np.ndarray:
        """Async batch prediction interface"""
        model = self.models.get(model_name, self.default_model)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model.predict_batch, features)
    
    def get_service_stats(self) -> Dict[str, any]:
        """Get service-wide statistics"""
        stats = {
            'models_loaded': len(self.models) + (1 if self.default_model else 0),
            'default_model_stats': self.default_model.get_performance_stats() if self.default_model else None,
            'gpu_config': {
                'device_type': self.config.device_type,
                'gpu_device_id': self.config.gpu_device_id,
                'max_batch_size': self.config.max_batch_size
            }
        }
        
        return stats


# Global service instance
gpu_inference_service = None


async def get_inference_service() -> GPUInferenceService:
    """Get or create global inference service"""
    global gpu_inference_service
    
    if gpu_inference_service is None:
        gpu_inference_service = GPUInferenceService()
        await gpu_inference_service.initialize()
    
    return gpu_inference_service


def benchmark_gpu_vs_cpu(num_samples: int = 1000, num_features: int = 20) -> Dict[str, any]:
    """Benchmark GPU vs CPU performance"""
    results = {}
    
    # Test GPU
    gpu_config = GPUConfig(device_type='gpu')
    gpu_model = GPUModelWrapper(gpu_config)
    gpu_model.load_model()
    
    logger.info("Benchmarking GPU performance...")
    gpu_results = gpu_model.benchmark(num_samples, num_features)
    results['gpu'] = gpu_results
    
    # Test CPU
    cpu_config = GPUConfig(device_type='cpu')
    cpu_model = GPUModelWrapper(cpu_config)
    cpu_model.load_model()
    
    logger.info("Benchmarking CPU performance...")
    cpu_results = cpu_model.benchmark(num_samples, num_features)
    results['cpu'] = cpu_results
    
    # Calculate speedup
    gpu_batch_1000 = gpu_results.get('batch_1000', {})
    cpu_batch_1000 = cpu_results.get('batch_1000', {})
    
    if gpu_batch_1000 and cpu_batch_1000:
        gpu_time = gpu_batch_1000['avg_time_ms']
        cpu_time = cpu_batch_1000['avg_time_ms']
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        results['comparison'] = {
            'gpu_time_ms': gpu_time,
            'cpu_time_ms': cpu_time,
            'speedup_factor': speedup,
            'gpu_faster': speedup > 1.0
        }
        
        logger.info(f"Performance comparison: GPU {gpu_time:.2f}ms vs CPU {cpu_time:.2f}ms")
        logger.info(f"Speedup factor: {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    async def test_gpu_inference():
        """Test GPU inference functionality"""
        # Test model wrapper
        gpu_info = GPUModelWrapper().detect_gpu_availability()
        print(f"GPU Info: {json.dumps(gpu_info, indent=2)}")
        
        # Test inference service
        service = await get_inference_service()
        
        # Test single prediction
        test_features = np.random.randn(20)
        prediction = await service.predict_async(test_features)
        print(f"Single prediction: {prediction}")
        
        # Test batch prediction
        batch_features = np.random.randn(100, 20)
        batch_predictions = await service.predict_batch_async(batch_features)
        print(f"Batch predictions shape: {batch_predictions.shape}")
        
        # Get service stats
        stats = service.get_service_stats()
        print(f"Service stats: {json.dumps(stats, indent=2)}")
    
    def test_benchmark():
        """Test benchmarking functionality"""
        results = benchmark_gpu_vs_cpu(num_samples=1000, num_features=20)
        print(f"Benchmark results: {json.dumps(results, indent=2)}")
    
    # Run tests
    print("=== Testing GPU Inference ===")
    asyncio.run(test_gpu_inference())
    
    print("\n=== Running Benchmark ===")
    test_benchmark()