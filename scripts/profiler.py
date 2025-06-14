#!/usr/bin/env python3
"""
Performance Profiling and FlameGraph Generator
Phase P11 Week 4 Day 4 - Performance analysis with py-spy and visualization

Features:
- Automated py-spy profiling of running processes
- FlameGraph generation and S3 upload
- Performance bottleneck detection
- Grafana dashboard integration
- Continuous profiling with scheduling
"""

import os
import time
import logging
import subprocess
import json
import argparse
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import psutil
import requests
import schedule

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    boto3 = None
    AWS_AVAILABLE = False

from prometheus_client import Counter, Histogram, Gauge, start_http_server

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('profiler')

# Prometheus metrics
profiling_sessions = Counter('profiling_sessions_total', 'Total profiling sessions', ['target', 'status'])
profiling_duration = Histogram('profiling_duration_seconds', 'Profiling session duration', ['target'])
flamegraph_generation = Counter('flamegraph_generation_total', 'FlameGraph generation attempts', ['status'])
performance_hotspots = Gauge('performance_hotspots_detected', 'Number of performance hotspots detected', ['process'])
cpu_usage_during_profiling = Gauge('cpu_usage_during_profiling_percent', 'CPU usage during profiling', ['process'])


class ProcessMonitor:
    """Monitor and identify target processes for profiling"""
    
    def __init__(self):
        self.target_processes = []
        self.process_cache = {}
        self.update_interval = 30  # seconds
    
    def find_mech_exo_processes(self) -> List[Dict]:
        """Find all Mech-Exo related processes"""
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    cmdline = ' '.join(proc_info['cmdline']) if proc_info['cmdline'] else ''
                    
                    # Look for Mech-Exo processes
                    if any(keyword in cmdline.lower() for keyword in ['mech_exo', 'mech-exo']):
                        processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'cmdline': cmdline,
                            'cpu_percent': proc_info['cpu_percent'],
                            'memory_percent': proc_info['memory_percent'],
                            'type': self._classify_process(cmdline)
                        })
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        
        except Exception as e:
            logger.error(f"Error finding processes: {e}")
        
        return processes
    
    def _classify_process(self, cmdline: str) -> str:
        """Classify process type based on command line"""
        cmdline_lower = cmdline.lower()
        
        if 'api' in cmdline_lower or 'uvicorn' in cmdline_lower:
            return 'api'
        elif 'exec' in cmdline_lower or 'execution' in cmdline_lower:
            return 'execution'
        elif 'dash' in cmdline_lower or 'dashboard' in cmdline_lower:
            return 'dashboard'
        elif 'ml' in cmdline_lower or 'model' in cmdline_lower:
            return 'ml'
        else:
            return 'unknown'
    
    def get_kubernetes_processes(self) -> List[Dict]:
        """Get processes from Kubernetes pods"""
        processes = []
        
        try:
            # Get Mech-Exo pods
            result = subprocess.run([
                'kubectl', 'get', 'pods', '-l', 'app.kubernetes.io/name=mech-exo',
                '-o', 'json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                
                for pod in pods_data.get('items', []):
                    pod_name = pod['metadata']['name']
                    namespace = pod['metadata']['namespace']
                    
                    # Get main process PID from pod
                    exec_result = subprocess.run([
                        'kubectl', 'exec', pod_name, '-n', namespace, '--',
                        'sh', '-c', 'ps aux | grep python | grep -v grep | head -1 | awk \'{print $2}\''
                    ], capture_output=True, text=True, timeout=10)
                    
                    if exec_result.returncode == 0 and exec_result.stdout.strip():
                        pid = exec_result.stdout.strip()
                        
                        processes.append({
                            'pod_name': pod_name,
                            'namespace': namespace,
                            'pid': pid,
                            'type': self._classify_process(pod_name),
                            'is_kubernetes': True
                        })
        
        except Exception as e:
            logger.warning(f"Failed to get Kubernetes processes: {e}")
        
        return processes


class PySpyProfiler:
    """Wrapper for py-spy profiling tool"""
    
    def __init__(self):
        self.py_spy_available = self._check_py_spy()
        self.default_duration = 60  # seconds
        self.default_rate = 100  # Hz
    
    def _check_py_spy(self) -> bool:
        """Check if py-spy is available"""
        try:
            result = subprocess.run(['py-spy', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def profile_process(self, 
                       pid: int, 
                       duration: int = None,
                       output_file: str = None,
                       rate: int = None) -> Optional[str]:
        """Profile a process and generate flame graph"""
        if not self.py_spy_available:
            logger.error("py-spy not available")
            return None
        
        duration = duration or self.default_duration
        rate = rate or self.default_rate
        
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'/tmp/flamegraph_{pid}_{timestamp}.svg'
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting profiling of PID {pid} for {duration}s")
            
            # Record profile data
            profile_data_file = output_file.replace('.svg', '.prof')
            
            record_cmd = [
                'py-spy', 'record',
                '-o', profile_data_file,
                '-d', str(duration),
                '-r', str(rate),
                '-p', str(pid)
            ]
            
            result = subprocess.run(record_cmd, capture_output=True, text=True, timeout=duration + 30)
            
            if result.returncode != 0:
                logger.error(f"py-spy record failed: {result.stderr}")
                profiling_sessions.labels(target=str(pid), status='failed').inc()
                return None
            
            # Generate flame graph
            flamegraph_cmd = [
                'py-spy', 'flamegraph',
                '-o', output_file,
                profile_data_file
            ]
            
            flamegraph_result = subprocess.run(flamegraph_cmd, capture_output=True, text=True, timeout=60)
            
            if flamegraph_result.returncode == 0:
                # Update metrics
                profile_duration = time.time() - start_time
                profiling_sessions.labels(target=str(pid), status='success').inc()
                profiling_duration.labels(target=str(pid)).observe(profile_duration)
                flamegraph_generation.labels(status='success').inc()
                
                logger.info(f"FlameGraph generated: {output_file}")
                return output_file
            else:
                logger.error(f"FlameGraph generation failed: {flamegraph_result.stderr}")
                flamegraph_generation.labels(status='failed').inc()
                return None
        
        except subprocess.TimeoutExpired:
            logger.error(f"Profiling timed out for PID {pid}")
            profiling_sessions.labels(target=str(pid), status='timeout').inc()
            return None
        except Exception as e:
            logger.error(f"Profiling error for PID {pid}: {e}")
            profiling_sessions.labels(target=str(pid), status='error').inc()
            return None
        finally:
            # Cleanup profile data file
            if 'profile_data_file' in locals() and Path(profile_data_file).exists():
                try:
                    os.remove(profile_data_file)
                except:
                    pass
    
    def profile_kubernetes_pod(self, 
                              pod_name: str, 
                              namespace: str = 'default',
                              duration: int = None,
                              output_file: str = None) -> Optional[str]:
        """Profile a process inside a Kubernetes pod"""
        if not self.py_spy_available:
            logger.error("py-spy not available")
            return None
        
        duration = duration or self.default_duration
        
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'/tmp/flamegraph_{pod_name}_{timestamp}.svg'
        
        try:
            logger.info(f"Starting profiling of pod {pod_name} for {duration}s")
            
            # Profile directly in the pod
            kubectl_cmd = [
                'kubectl', 'exec', pod_name, '-n', namespace, '--',
                'py-spy', 'record',
                '-o', '/tmp/profile.svg',
                '-d', str(duration),
                '-r', str(self.default_rate),
                '--pid', '1'  # Usually the main process in container
            ]
            
            result = subprocess.run(kubectl_cmd, capture_output=True, text=True, timeout=duration + 60)
            
            if result.returncode == 0:
                # Copy flame graph from pod
                copy_cmd = [
                    'kubectl', 'cp', 
                    f'{namespace}/{pod_name}:/tmp/profile.svg',
                    output_file
                ]
                
                copy_result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=30)
                
                if copy_result.returncode == 0:
                    profiling_sessions.labels(target=pod_name, status='success').inc()
                    flamegraph_generation.labels(status='success').inc()
                    logger.info(f"Pod FlameGraph generated: {output_file}")
                    return output_file
                else:
                    logger.error(f"Failed to copy FlameGraph from pod: {copy_result.stderr}")
            else:
                logger.error(f"Pod profiling failed: {result.stderr}")
            
            profiling_sessions.labels(target=pod_name, status='failed').inc()
            return None
            
        except Exception as e:
            logger.error(f"Pod profiling error: {e}")
            profiling_sessions.labels(target=pod_name, status='error').inc()
            return None


class S3Uploader:
    """Upload flame graphs to S3 for storage and sharing"""
    
    def __init__(self, bucket_name: str = 'mech-exo-reports', region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = None
        
        if AWS_AVAILABLE:
            try:
                self.s3_client = boto3.client('s3', region_name=region)
                logger.info(f"S3 uploader initialized for bucket: {bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}")
    
    def upload_flamegraph(self, local_file: str, s3_key: str = None) -> Optional[str]:
        """Upload flame graph to S3"""
        if not self.s3_client:
            logger.warning("S3 client not available")
            return None
        
        if not Path(local_file).exists():
            logger.error(f"Local file not found: {local_file}")
            return None
        
        if not s3_key:
            filename = Path(local_file).name
            date_str = datetime.now().strftime('%Y/%m/%d')
            s3_key = f'flamegraphs/{date_str}/{filename}'
        
        try:
            self.s3_client.upload_file(
                local_file, 
                self.bucket_name, 
                s3_key,
                ExtraArgs={'ContentType': 'image/svg+xml'}
            )
            
            # Generate public URL
            s3_url = f'https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}'
            
            logger.info(f"FlameGraph uploaded to S3: {s3_url}")
            return s3_url
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return None
    
    def list_recent_flamegraphs(self, days: int = 7) -> List[Dict]:
        """List recent flame graphs from S3"""
        if not self.s3_client:
            return []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='flamegraphs/'
            )
            
            flamegraphs = []
            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) > cutoff_date:
                    flamegraphs.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'url': f'https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{obj["Key"]}'
                    })
            
            return sorted(flamegraphs, key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []


class PerformanceAnalyzer:
    """Analyze flame graphs and detect performance issues"""
    
    def __init__(self):
        self.hotspot_patterns = [
            'blocking_io',
            'network_wait',
            'database_query',
            'serialization',
            'computation_heavy',
            'memory_allocation'
        ]
    
    def analyze_flamegraph(self, svg_file: str) -> Dict[str, any]:
        """Analyze flame graph and extract performance insights"""
        if not Path(svg_file).exists():
            return {}
        
        try:
            # Read SVG content
            with open(svg_file, 'r') as f:
                svg_content = f.read()
            
            # Simple text-based analysis (in production, use more sophisticated parsing)
            hotspots = self._detect_hotspots(svg_content)
            recommendations = self._generate_recommendations(hotspots)
            
            analysis = {
                'file': svg_file,
                'timestamp': datetime.now().isoformat(),
                'hotspots': hotspots,
                'recommendations': recommendations,
                'top_functions': self._extract_top_functions(svg_content)
            }
            
            # Update metrics
            performance_hotspots.labels(process=Path(svg_file).stem).set(len(hotspots))
            
            return analysis
            
        except Exception as e:
            logger.error(f"FlameGraph analysis failed: {e}")
            return {}
    
    def _detect_hotspots(self, svg_content: str) -> List[Dict]:
        """Detect performance hotspots in flame graph"""
        hotspots = []
        
        # Look for common performance bottleneck patterns
        patterns = {
            'io_operations': ['read', 'write', 'recv', 'send', 'socket'],
            'database_calls': ['query', 'execute', 'fetch', 'connection'],
            'serialization': ['json', 'pickle', 'serialize', 'marshal'],
            'computation': ['calculate', 'compute', 'process', 'transform'],
            'memory_ops': ['malloc', 'alloc', 'gc', 'garbage']
        }
        
        for category, keywords in patterns.items():
            for keyword in keywords:
                if keyword in svg_content.lower():
                    hotspots.append({
                        'category': category,
                        'pattern': keyword,
                        'severity': 'medium'  # Simplified scoring
                    })
        
        return hotspots
    
    def _generate_recommendations(self, hotspots: List[Dict]) -> List[str]:
        """Generate optimization recommendations based on hotspots"""
        recommendations = []
        categories = {h['category'] for h in hotspots}
        
        if 'io_operations' in categories:
            recommendations.append("Consider using async I/O operations")
            recommendations.append("Implement connection pooling")
        
        if 'database_calls' in categories:
            recommendations.append("Optimize database queries")
            recommendations.append("Add query result caching")
        
        if 'serialization' in categories:
            recommendations.append("Use faster serialization formats (msgpack, protobuf)")
            recommendations.append("Cache serialized objects")
        
        if 'computation' in categories:
            recommendations.append("Profile computational bottlenecks")
            recommendations.append("Consider parallel processing")
        
        if 'memory_ops' in categories:
            recommendations.append("Optimize memory usage patterns")
            recommendations.append("Reduce object allocations")
        
        return recommendations
    
    def _extract_top_functions(self, svg_content: str) -> List[str]:
        """Extract top functions from flame graph (simplified)"""
        # This is a simplified extraction - in production, parse SVG properly
        functions = []
        
        # Look for function names in SVG text elements
        import re
        pattern = r'<text[^>]*>([^<]+)</text>'
        matches = re.findall(pattern, svg_content)
        
        # Filter and deduplicate
        seen = set()
        for match in matches:
            if '(' in match and len(match) > 5 and match not in seen:
                functions.append(match.strip())
                seen.add(match)
                if len(functions) >= 10:  # Top 10
                    break
        
        return functions


class ProfilerManager:
    """Main profiler manager coordinating all components"""
    
    def __init__(self, 
                 enable_s3: bool = True,
                 enable_kubernetes: bool = True,
                 profiling_duration: int = 60):
        self.process_monitor = ProcessMonitor()
        self.py_spy_profiler = PySpyProfiler()
        self.s3_uploader = S3Uploader() if enable_s3 and AWS_AVAILABLE else None
        self.performance_analyzer = PerformanceAnalyzer()
        
        self.enable_kubernetes = enable_kubernetes
        self.profiling_duration = profiling_duration
        self.results_directory = Path('/tmp/profiling_results')
        self.results_directory.mkdir(exist_ok=True)
        
        # Metrics server
        self.metrics_port = 9093
    
    def profile_all_targets(self) -> List[Dict]:
        """Profile all identified target processes"""
        results = []
        
        # Profile local processes
        processes = self.process_monitor.find_mech_exo_processes()
        
        for proc in processes:
            try:
                logger.info(f"Profiling process: {proc['name']} (PID: {proc['pid']})")
                
                flamegraph_path = self.py_spy_profiler.profile_process(
                    proc['pid'], 
                    self.profiling_duration
                )
                
                if flamegraph_path:
                    # Analyze flame graph
                    analysis = self.performance_analyzer.analyze_flamegraph(flamegraph_path)
                    
                    # Upload to S3 if available
                    s3_url = None
                    if self.s3_uploader:
                        s3_url = self.s3_uploader.upload_flamegraph(flamegraph_path)
                    
                    results.append({
                        'process': proc,
                        'flamegraph_path': flamegraph_path,
                        's3_url': s3_url,
                        'analysis': analysis,
                        'timestamp': datetime.now().isoformat()
                    })
            
            except Exception as e:
                logger.error(f"Failed to profile process {proc['pid']}: {e}")
        
        # Profile Kubernetes pods if enabled
        if self.enable_kubernetes:
            k8s_processes = self.process_monitor.get_kubernetes_processes()
            
            for proc in k8s_processes:
                try:
                    logger.info(f"Profiling pod: {proc['pod_name']}")
                    
                    flamegraph_path = self.py_spy_profiler.profile_kubernetes_pod(
                        proc['pod_name'],
                        proc['namespace'],
                        self.profiling_duration
                    )
                    
                    if flamegraph_path:
                        analysis = self.performance_analyzer.analyze_flamegraph(flamegraph_path)
                        
                        s3_url = None
                        if self.s3_uploader:
                            s3_url = self.s3_uploader.upload_flamegraph(flamegraph_path)
                        
                        results.append({
                            'process': proc,
                            'flamegraph_path': flamegraph_path,
                            's3_url': s3_url,
                            'analysis': analysis,
                            'timestamp': datetime.now().isoformat()
                        })
                
                except Exception as e:
                    logger.error(f"Failed to profile pod {proc['pod_name']}: {e}")
        
        return results
    
    def generate_profiling_report(self, results: List[Dict]) -> str:
        """Generate comprehensive profiling report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_directory / f'profiling_report_{timestamp}.json'
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'profiling_duration': self.profiling_duration,
                'total_targets': len(results),
                'successful_profiles': len([r for r in results if r['flamegraph_path']])
            },
            'results': results,
            'summary': self._generate_summary(results)
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Profiling report generated: {report_file}")
        return str(report_file)
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary of profiling results"""
        successful_profiles = [r for r in results if r['flamegraph_path']]
        
        all_hotspots = []
        all_recommendations = set()
        
        for result in successful_profiles:
            analysis = result.get('analysis', {})
            all_hotspots.extend(analysis.get('hotspots', []))
            all_recommendations.update(analysis.get('recommendations', []))
        
        return {
            'total_profiles': len(results),
            'successful_profiles': len(successful_profiles),
            'total_hotspots': len(all_hotspots),
            'hotspot_categories': list(set(h['category'] for h in all_hotspots)),
            'optimization_recommendations': list(all_recommendations),
            's3_uploads': len([r for r in successful_profiles if r['s3_url']])
        }
    
    def run_scheduled_profiling(self, interval_hours: int = 6):
        """Run profiling on a schedule"""
        logger.info(f"Starting scheduled profiling every {interval_hours} hours")
        
        def run_profiling():
            logger.info("Running scheduled profiling session")
            results = self.profile_all_targets()
            self.generate_profiling_report(results)
        
        # Initial run
        run_profiling()
        
        # Schedule future runs
        schedule.every(interval_hours).hours.do(run_profiling)
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def run(self):
        """Run the profiler manager"""
        logger.info("Starting Performance Profiler Manager")
        
        # Start metrics server
        start_http_server(self.metrics_port)
        logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        
        try:
            # Run one-time profiling
            results = self.profile_all_targets()
            report_file = self.generate_profiling_report(results)
            
            logger.info(f"Profiling completed. Report: {report_file}")
            
            # Print summary
            summary = self._generate_summary(results)
            print("\n" + "="*60)
            print("PERFORMANCE PROFILING SUMMARY")
            print("="*60)
            print(f"Total profiles: {summary['total_profiles']}")
            print(f"Successful: {summary['successful_profiles']}")
            print(f"Hotspots detected: {summary['total_hotspots']}")
            print(f"Categories: {', '.join(summary['hotspot_categories'])}")
            print(f"S3 uploads: {summary['s3_uploads']}")
            print("\nTop Recommendations:")
            for rec in summary['optimization_recommendations'][:5]:
                print(f"  - {rec}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Performance Profiler and FlameGraph Generator')
    parser.add_argument('--duration', type=int, default=60, help='Profiling duration in seconds')
    parser.add_argument('--no-s3', action='store_true', help='Disable S3 uploads')
    parser.add_argument('--no-k8s', action='store_true', help='Disable Kubernetes profiling')
    parser.add_argument('--scheduled', type=int, help='Run scheduled profiling every N hours')
    parser.add_argument('--target-pid', type=int, help='Profile specific PID')
    parser.add_argument('--target-pod', type=str, help='Profile specific pod')
    
    args = parser.parse_args()
    
    profiler = ProfilerManager(
        enable_s3=not args.no_s3,
        enable_kubernetes=not args.no_k8s,
        profiling_duration=args.duration
    )
    
    if args.target_pid:
        # Profile specific PID
        flamegraph = profiler.py_spy_profiler.profile_process(args.target_pid, args.duration)
        if flamegraph:
            print(f"FlameGraph generated: {flamegraph}")
        
    elif args.target_pod:
        # Profile specific pod
        flamegraph = profiler.py_spy_profiler.profile_kubernetes_pod(args.target_pod, duration=args.duration)
        if flamegraph:
            print(f"Pod FlameGraph generated: {flamegraph}")
    
    elif args.scheduled:
        # Run scheduled profiling
        profiler.run_scheduled_profiling(args.scheduled)
    
    else:
        # Run one-time profiling of all targets
        profiler.run()


if __name__ == "__main__":
    main()