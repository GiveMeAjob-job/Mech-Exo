#!/usr/bin/env python3
"""
GPU Inference Benchmark Script
Phase P11 Week 4 Day 2 - Performance benchmarking for LightGBM GPU vs CPU

Usage:
    python ml/bench_gpu.py --n 1000 --features 20
    python ml/bench_gpu.py --batch-sizes 1,10,100,1000 --trials 5
"""

import argparse
import json
import time
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.gpu_infer import GPUModelWrapper, GPUConfig, benchmark_gpu_vs_cpu

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Advanced benchmarking for GPU inference"""
    
    def __init__(self):
        self.results = {}
        self.timestamp = time.time()
    
    def run_comprehensive_benchmark(self, 
                                   sample_counts: List[int] = None,
                                   feature_counts: List[int] = None,
                                   batch_sizes: List[int] = None,
                                   trials: int = 5) -> Dict[str, Any]:
        """Run comprehensive benchmark across different configurations"""
        
        sample_counts = sample_counts or [100, 500, 1000, 5000]
        feature_counts = feature_counts or [10, 20, 50, 100]
        batch_sizes = batch_sizes or [1, 10, 50, 100, 500, 1000]
        
        logger.info("Starting comprehensive GPU vs CPU benchmark")
        logger.info(f"Sample counts: {sample_counts}")
        logger.info(f"Feature counts: {feature_counts}")
        logger.info(f"Batch sizes: {batch_sizes}")
        logger.info(f"Trials per configuration: {trials}")
        
        results = {
            'metadata': {
                'timestamp': self.timestamp,
                'trials': trials,
                'configurations': len(sample_counts) * len(feature_counts)
            },
            'results': {},
            'summary': {}
        }
        
        for num_samples in sample_counts:
            for num_features in feature_counts:
                config_name = f"samples_{num_samples}_features_{num_features}"
                logger.info(f"Benchmarking configuration: {config_name}")
                
                config_results = self._benchmark_configuration(
                    num_samples, num_features, batch_sizes, trials
                )
                
                results['results'][config_name] = config_results
        
        # Generate summary
        results['summary'] = self._generate_summary(results['results'])
        
        return results
    
    def _benchmark_configuration(self, 
                                num_samples: int, 
                                num_features: int,
                                batch_sizes: List[int],
                                trials: int) -> Dict[str, Any]:
        """Benchmark a specific configuration"""
        
        # Generate test data
        np.random.seed(42)
        test_data = np.random.randn(num_samples, num_features)
        
        config_results = {
            'gpu': {},
            'cpu': {},
            'comparison': {}
        }
        
        # Test GPU
        try:
            gpu_config = GPUConfig(device_type='gpu')
            gpu_model = GPUModelWrapper(gpu_config)
            gpu_loaded = gpu_model.load_model()
            
            if gpu_loaded:
                config_results['gpu'] = self._benchmark_device(
                    gpu_model, test_data, batch_sizes, trials
                )
            else:
                logger.warning("GPU model failed to load")
                config_results['gpu'] = {'error': 'Failed to load GPU model'}
        except Exception as e:
            logger.error(f"GPU benchmark error: {e}")
            config_results['gpu'] = {'error': str(e)}
        
        # Test CPU
        try:
            cpu_config = GPUConfig(device_type='cpu')
            cpu_model = GPUModelWrapper(cpu_config)
            cpu_loaded = cpu_model.load_model()
            
            if cpu_loaded:
                config_results['cpu'] = self._benchmark_device(
                    cpu_model, test_data, batch_sizes, trials
                )
            else:
                logger.warning("CPU model failed to load")
                config_results['cpu'] = {'error': 'Failed to load CPU model'}
        except Exception as e:
            logger.error(f"CPU benchmark error: {e}")
            config_results['cpu'] = {'error': str(e)}
        
        # Compare results
        if 'error' not in config_results['gpu'] and 'error' not in config_results['cpu']:
            config_results['comparison'] = self._compare_results(
                config_results['gpu'], config_results['cpu']
            )
        
        return config_results
    
    def _benchmark_device(self, 
                         model: GPUModelWrapper,
                         test_data: np.ndarray,
                         batch_sizes: List[int],
                         trials: int) -> Dict[str, Any]:
        """Benchmark a specific device (GPU or CPU)"""
        
        device_results = {
            'device_type': model.device_type,
            'batch_results': {},
            'overall_stats': {}
        }
        
        num_samples = len(test_data)
        
        for batch_size in batch_sizes:
            if batch_size > num_samples:
                continue
            
            batch_times = []
            throughputs = []
            
            # Run multiple trials
            for trial in range(trials):
                start_idx = (trial * batch_size) % (num_samples - batch_size)
                end_idx = start_idx + batch_size
                batch_data = test_data[start_idx:end_idx]
                
                # Warmup run
                if trial == 0:
                    _ = model.predict_batch(batch_data)
                
                # Measured run
                start_time = time.time()
                predictions = model.predict_batch(batch_data)
                end_time = time.time()
                
                batch_time = end_time - start_time
                throughput = batch_size / batch_time
                
                batch_times.append(batch_time)
                throughputs.append(throughput)
            
            # Calculate statistics
            avg_time = np.mean(batch_times)
            std_time = np.std(batch_times)
            min_time = np.min(batch_times)
            max_time = np.max(batch_times)
            
            avg_throughput = np.mean(throughputs)
            latency_per_sample = avg_time / batch_size
            
            device_results['batch_results'][f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'avg_throughput_per_second': avg_throughput,
                'latency_per_sample_ms': latency_per_sample * 1000,
                'trials': trials
            }
        
        # Overall device statistics
        all_times = []
        all_throughputs = []
        
        for batch_result in device_results['batch_results'].values():
            all_times.append(batch_result['avg_time_ms'])
            all_throughputs.append(batch_result['avg_throughput_per_second'])
        
        if all_times:
            device_results['overall_stats'] = {
                'best_latency_ms': min(all_times),
                'worst_latency_ms': max(all_times),
                'best_throughput_per_second': max(all_throughputs),
                'avg_throughput_per_second': np.mean(all_throughputs)
            }
        
        # Get model performance stats
        perf_stats = model.get_performance_stats()
        device_results['model_stats'] = perf_stats
        
        return device_results
    
    def _compare_results(self, gpu_results: Dict, cpu_results: Dict) -> Dict[str, Any]:
        """Compare GPU vs CPU results"""
        comparison = {
            'speedup_by_batch': {},
            'overall_comparison': {}
        }
        
        # Compare by batch size
        gpu_batches = gpu_results.get('batch_results', {})
        cpu_batches = cpu_results.get('batch_results', {})
        
        for batch_key in gpu_batches.keys():
            if batch_key in cpu_batches:
                gpu_time = gpu_batches[batch_key]['avg_time_ms']
                cpu_time = cpu_batches[batch_key]['avg_time_ms']
                
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                comparison['speedup_by_batch'][batch_key] = {
                    'gpu_time_ms': gpu_time,
                    'cpu_time_ms': cpu_time,
                    'speedup_factor': speedup,
                    'gpu_faster': speedup > 1.0,
                    'time_saved_ms': cpu_time - gpu_time,
                    'time_saved_percent': ((cpu_time - gpu_time) / cpu_time * 100) if cpu_time > 0 else 0
                }
        
        # Overall comparison
        gpu_overall = gpu_results.get('overall_stats', {})
        cpu_overall = cpu_results.get('overall_stats', {})
        
        if gpu_overall and cpu_overall:
            gpu_best_latency = gpu_overall.get('best_latency_ms', float('inf'))
            cpu_best_latency = cpu_overall.get('best_latency_ms', float('inf'))
            
            gpu_best_throughput = gpu_overall.get('best_throughput_per_second', 0)
            cpu_best_throughput = cpu_overall.get('best_throughput_per_second', 0)
            
            comparison['overall_comparison'] = {
                'latency_speedup': cpu_best_latency / gpu_best_latency if gpu_best_latency > 0 else 0,
                'throughput_improvement': gpu_best_throughput / cpu_best_throughput if cpu_best_throughput > 0 else 0,
                'gpu_best_latency_ms': gpu_best_latency,
                'cpu_best_latency_ms': cpu_best_latency,
                'gpu_best_throughput': gpu_best_throughput,
                'cpu_best_throughput': cpu_best_throughput
            }
        
        return comparison
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all benchmark results"""
        summary = {
            'configurations_tested': len(results),
            'gpu_performance': {},
            'cpu_performance': {},
            'speedup_analysis': {}
        }
        
        gpu_speedups = []
        cpu_times = []
        gpu_times = []
        
        # Aggregate results across all configurations
        for config_name, config_results in results.items():
            comparison = config_results.get('comparison', {})
            
            if 'overall_comparison' in comparison:
                overall = comparison['overall_comparison']
                
                if 'latency_speedup' in overall:
                    gpu_speedups.append(overall['latency_speedup'])
                
                if 'gpu_best_latency_ms' in overall:
                    gpu_times.append(overall['gpu_best_latency_ms'])
                
                if 'cpu_best_latency_ms' in overall:
                    cpu_times.append(overall['cpu_best_latency_ms'])
        
        # Calculate summary statistics
        if gpu_speedups:
            summary['speedup_analysis'] = {
                'avg_speedup_factor': np.mean(gpu_speedups),
                'max_speedup_factor': np.max(gpu_speedups),
                'min_speedup_factor': np.min(gpu_speedups),
                'speedup_std': np.std(gpu_speedups),
                'configurations_where_gpu_faster': sum(1 for s in gpu_speedups if s > 1.0),
                'gpu_advantage_percentage': (sum(1 for s in gpu_speedups if s > 1.0) / len(gpu_speedups)) * 100
            }
        
        if gpu_times:
            summary['gpu_performance'] = {
                'avg_best_latency_ms': np.mean(gpu_times),
                'best_latency_ms': np.min(gpu_times),
                'worst_latency_ms': np.max(gpu_times)
            }
        
        if cpu_times:
            summary['cpu_performance'] = {
                'avg_best_latency_ms': np.mean(cpu_times),
                'best_latency_ms': np.min(cpu_times),
                'worst_latency_ms': np.max(cpu_times)
            }
        
        # Performance targets
        target_gpu_latency = 25  # ms
        target_speedup = 3.0
        
        summary['target_analysis'] = {
            'target_gpu_latency_ms': target_gpu_latency,
            'target_speedup_factor': target_speedup,
            'gpu_latency_target_met': min(gpu_times) < target_gpu_latency if gpu_times else False,
            'speedup_target_met': max(gpu_speedups) >= target_speedup if gpu_speedups else False
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """Save benchmark results to file"""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"benchmark_results_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {output_path}")
        return output_path
    
    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary to console"""
        summary = results.get('summary', {})
        
        print("\n" + "="*60)
        print("GPU INFERENCE BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Configurations tested: {summary.get('configurations_tested', 0)}")
        
        speedup_analysis = summary.get('speedup_analysis', {})
        if speedup_analysis:
            print(f"\nSpeedup Analysis:")
            print(f"  Average speedup: {speedup_analysis.get('avg_speedup_factor', 0):.2f}x")
            print(f"  Maximum speedup: {speedup_analysis.get('max_speedup_factor', 0):.2f}x")
            print(f"  GPU faster in: {speedup_analysis.get('gpu_advantage_percentage', 0):.1f}% of cases")
        
        gpu_perf = summary.get('gpu_performance', {})
        cpu_perf = summary.get('cpu_performance', {})
        
        if gpu_perf:
            print(f"\nGPU Performance:")
            print(f"  Best latency: {gpu_perf.get('best_latency_ms', 0):.2f}ms")
            print(f"  Average best latency: {gpu_perf.get('avg_best_latency_ms', 0):.2f}ms")
        
        if cpu_perf:
            print(f"\nCPU Performance:")
            print(f"  Best latency: {cpu_perf.get('best_latency_ms', 0):.2f}ms")
            print(f"  Average best latency: {cpu_perf.get('avg_best_latency_ms', 0):.2f}ms")
        
        target_analysis = summary.get('target_analysis', {})
        if target_analysis:
            print(f"\nTarget Analysis:")
            print(f"  GPU latency target (<{target_analysis.get('target_gpu_latency_ms', 0)}ms): {'✅ MET' if target_analysis.get('gpu_latency_target_met', False) else '❌ NOT MET'}")
            print(f"  Speedup target (≥{target_analysis.get('target_speedup_factor', 0)}x): {'✅ MET' if target_analysis.get('speedup_target_met', False) else '❌ NOT MET'}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='GPU Inference Benchmark')
    parser.add_argument('--n', type=int, default=1000, help='Number of samples (default: 1000)')
    parser.add_argument('--features', type=int, default=20, help='Number of features (default: 20)')
    parser.add_argument('--batch-sizes', type=str, default='1,10,50,100,500,1000', 
                       help='Comma-separated batch sizes (default: 1,10,50,100,500,1000)')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per configuration (default: 5)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Run comprehensive benchmark across multiple configurations')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    
    runner = BenchmarkRunner()
    
    if args.comprehensive:
        # Comprehensive benchmark
        sample_counts = [100, 500, 1000, 2000] if not args.quick else [1000]
        feature_counts = [10, 20, 50] if not args.quick else [20]
        
        results = runner.run_comprehensive_benchmark(
            sample_counts=sample_counts,
            feature_counts=feature_counts,
            batch_sizes=batch_sizes,
            trials=args.trials
        )
    else:
        # Simple benchmark
        logger.info(f"Running simple benchmark: {args.n} samples, {args.features} features")
        simple_results = benchmark_gpu_vs_cpu(args.n, args.features)
        
        results = {
            'metadata': {'timestamp': time.time(), 'type': 'simple'},
            'results': {'simple_benchmark': simple_results},
            'summary': {}
        }
    
    # Print summary
    runner.print_summary(results)
    
    # Save results
    if args.output:
        runner.save_results(results, args.output)
    else:
        output_file = runner.save_results(results)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()