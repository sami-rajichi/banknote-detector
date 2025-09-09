#!/usr/bin/env python3
"""
Find optimal number of workers for YOLO training
Tests different worker counts and measures performance
"""

import time
import psutil
import torch
import gc
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

def get_system_specs():
    """Get basic system specifications"""
    specs = {
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        'cpu_cores_logical': psutil.cpu_count(logical=True),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU",
        'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    }
    
    print("🖥️  SYSTEM SPECIFICATIONS")
    print("=" * 50)
    for key, value in specs.items():
        if 'gb' in key:
            print(f"{key.replace('_', ' ').title()}: {value:.1f} GB")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    print()
    
    return specs

def calculate_theoretical_max_workers(specs):
    """Calculate theoretical maximum workers based on system specs"""
    print("🧮 THEORETICAL LIMITS")
    print("=" * 50)
    
    # Rule of thumb calculations
    cpu_based_max = specs['cpu_cores_logical']  # Usually 1 worker per logical core
    ram_based_max = int(specs['ram_gb'] / 2)    # ~2GB RAM per worker (rough estimate)
    
    # Conservative recommendations
    conservative_max = min(specs['cpu_cores_physical'], 8)  # Usually don't exceed physical cores
    
    print(f"CPU-based limit: {cpu_based_max} workers (1 per logical core)")
    print(f"RAM-based limit: {ram_based_max} workers (~2GB RAM per worker)")
    print(f"Conservative recommendation: {conservative_max} workers")
    print(f"Suggested test range: 0 to {min(cpu_based_max, 12)}")
    print()
    
    return {
        'cpu_max': cpu_based_max,
        'ram_max': ram_based_max,
        'conservative': conservative_max,
        'test_max': min(cpu_based_max, 12)
    }

def benchmark_workers(worker_counts, dataset_path="./annotated_banknote_dataset/data.yaml", epochs=1):
    """
    Benchmark different worker counts
    """
    print("⚡ WORKER BENCHMARKING")
    print("=" * 50)
    
    results = {
        'workers': [],
        'time_per_epoch': [],
        'iterations_per_sec': [],
        'cpu_usage': [],
        'ram_usage': [],
        'gpu_memory': [],
        'status': []
    }
    
    for workers in worker_counts:
        print(f"\\n🧪 Testing workers={workers}")
        
        # Clear GPU memory before each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Record initial system state
            initial_ram = psutil.virtual_memory().percent
            initial_gpu = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            # Load model
            model = YOLO("yolo11n.pt")  # Use small model for testing
            
            # Start training
            start_time = time.time()
            
            # Monitor system during training
            max_cpu = 0
            max_ram = 0
            
            def monitor_system():
                nonlocal max_cpu, max_ram
                max_cpu = max(max_cpu, psutil.cpu_percent())
                max_ram = max(max_ram, psutil.virtual_memory().percent)
            
            # Quick training test
            model.train(
                data=dataset_path,
                epochs=epochs,
                workers=workers,
                batch=8,  # Fixed batch size for comparison
                imgsz=640,
                cache=False,  # Consistent cache setting
                verbose=False,
                plots=False,
                name=f"worker_test_{workers}",
                exist_ok=True
            )
            
            end_time = time.time()
            
            # Calculate metrics
            epoch_time = end_time - start_time
            iterations_per_sec = 1.0 / (epoch_time / 100) if epoch_time > 0 else 0  # Rough estimate
            
            # Record final system state
            final_ram = psutil.virtual_memory().percent
            final_gpu = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            # Store results
            results['workers'].append(workers)
            results['time_per_epoch'].append(epoch_time)
            results['iterations_per_sec'].append(iterations_per_sec)
            results['cpu_usage'].append(max_cpu)
            results['ram_usage'].append(final_ram)
            results['gpu_memory'].append(final_gpu)
            results['status'].append('SUCCESS')
            
            print(f"  ✅ Success: {epoch_time:.1f}s per epoch, ~{iterations_per_sec:.1f} it/s")
            print(f"     CPU: {max_cpu:.1f}%, RAM: {final_ram:.1f}%, GPU: {final_gpu:.1f}GB")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            
            # Still record the failure
            results['workers'].append(workers)
            results['time_per_epoch'].append(float('inf'))
            results['iterations_per_sec'].append(0)
            results['cpu_usage'].append(0)
            results['ram_usage'].append(0)
            results['gpu_memory'].append(0)
            results['status'].append(f'FAILED: {str(e)[:50]}')
    
    return results

def analyze_results(results):
    """Analyze benchmark results and find optimal worker count"""
    print("\\n📊 RESULTS ANALYSIS")
    print("=" * 50)
    
    # Find successful runs
    successful_results = []
    for i, status in enumerate(results['status']):
        if status == 'SUCCESS':
            successful_results.append({
                'workers': results['workers'][i],
                'time': results['time_per_epoch'][i],
                'it_per_sec': results['iterations_per_sec'][i],
                'cpu': results['cpu_usage'][i],
                'ram': results['ram_usage'][i],
                'gpu': results['gpu_memory'][i]
            })
    
    if not successful_results:
        print("❌ No successful runs found!")
        return None
    
    # Sort by performance (lowest time = best)
    successful_results.sort(key=lambda x: x['time'])
    
    print("Top performing configurations:")
    print(f"{'Workers':<8} {'Time(s)':<8} {'it/s':<8} {'CPU%':<8} {'RAM%':<8} {'GPU(GB)':<8}")
    print("-" * 60)
    
    for result in successful_results[:5]:  # Top 5
        print(f"{result['workers']:<8} {result['time']:<8.1f} {result['it_per_sec']:<8.1f} "
              f"{result['cpu']:<8.1f} {result['ram']:<8.1f} {result['gpu']:<8.1f}")
    
    # Find optimal worker count
    best_result = successful_results[0]
    
    print(f"\\n🎯 OPTIMAL CONFIGURATION:")
    print(f"   Workers: {best_result['workers']}")
    print(f"   Time per epoch: {best_result['time']:.1f}s")
    print(f"   Iterations/sec: {best_result['it_per_sec']:.1f}")
    print(f"   CPU usage: {best_result['cpu']:.1f}%")
    print(f"   RAM usage: {best_result['ram']:.1f}%")
    
    # Warnings
    if best_result['cpu'] > 90:
        print("   ⚠️  Warning: High CPU usage - consider reducing workers")
    if best_result['ram'] > 85:
        print("   ⚠️  Warning: High RAM usage - monitor memory")
    
    return best_result

def quick_worker_test(max_workers=None):
    """Quick test to find approximate optimal workers"""
    specs = get_system_specs()
    limits = calculate_theoretical_max_workers(specs)
    
    if max_workers is None:
        max_workers = limits['test_max']
    
    # Test range: 0, 1, 2, 4, 6, 8, 10, 12...
    worker_counts = [0, 1, 2, 4]
    
    # Add more workers up to limit
    current = 6
    while current <= max_workers:
        worker_counts.append(current)
        current += 2
    
    print(f"Testing worker counts: {worker_counts}")
    print("This will run a short training test for each worker count...\\n")
    
    # Run benchmark
    results = benchmark_workers(worker_counts)
    
    # Analyze results
    optimal = analyze_results(results)
    
    return optimal, results

def advanced_worker_tuning(dataset_path, batch_sizes=[4, 8, 16], image_sizes=[640, 832]):
    """
    Advanced tuning that tests workers with different batch sizes and image sizes
    """
    print("🔬 ADVANCED WORKER TUNING")
    print("=" * 50)
    print("Testing combinations of workers, batch sizes, and image sizes...")
    
    specs = get_system_specs()
    limits = calculate_theoretical_max_workers(specs)
    
    # Worker range to test
    worker_range = [4, 6, 8, 10, 12]
    
    best_configs = []
    
    for imgsz in image_sizes[:1]:
        for batch in [batch_sizes[-1]]:
            print(f"\\n📐 Testing imgsz={imgsz}, batch={batch}")
            
            for workers in worker_range:
                print(f"  Testing workers={workers}...", end=" ")
                
                try:
                    # Clear memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    model = YOLO("yolo11n.pt")
                    
                    start_time = time.time()
                    
                    model.train(
                        data=dataset_path,
                        epochs=1,
                        workers=workers,
                        batch=batch,
                        imgsz=imgsz,
                        cache=False,
                        verbose=False,
                        plots=False,
                        name=f"advanced_test_{imgsz}_{batch}_{workers}",
                        exist_ok=True
                    )
                    
                    epoch_time = time.time() - start_time
                    it_per_sec = 1.0 / (epoch_time / 100)
                    
                    best_configs.append({
                        'workers': workers,
                        'batch': batch,
                        'imgsz': imgsz,
                        'time': epoch_time,
                        'it_per_sec': it_per_sec,
                        'config_name': f"w{workers}_b{batch}_i{imgsz}"
                    })
                    
                    print(f"✅ {epoch_time:.1f}s ({it_per_sec:.1f} it/s)")
                    
                except Exception as e:
                    print(f"❌ Failed: {str(e)[:30]}")
    
    # Find best overall configuration
    best_configs.sort(key=lambda x: x['time'])
    
    print("\\n🏆 TOP CONFIGURATIONS:")
    print(f"{'Config':<15} {'Workers':<8} {'Batch':<6} {'ImgSz':<6} {'Time(s)':<8} {'it/s':<8}")
    print("-" * 65)
    
    for config in best_configs[:10]:
        print(f"{config['config_name']:<15} {config['workers']:<8} {config['batch']:<6} "
              f"{config['imgsz']:<6} {config['time']:<8.1f} {config['it_per_sec']:<8.1f}")
    
    return best_configs

def main():
    """Main function to find optimal workers"""
    print("🔍 YOLO WORKER OPTIMIZATION TOOL")
    print("=" * 70)
    
    # Choice of test type
    print("Choose test type:")
    print("1. Quick test (recommended)")
    print("2. Advanced test (comprehensive)")
    print("3. Custom worker range")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\\n🚀 Running quick worker optimization...")
        optimal, results = quick_worker_test()
        
        if optimal:
            print(f"\\n✅ RECOMMENDATION: Use workers={optimal['workers']} for your setup")
        
    elif choice == "2":
        dataset_path = input("Enter dataset path (or press Enter for default): ").strip()
        if not dataset_path:
            dataset_path = "./annotated_banknote_dataset/data.yaml"
        
        print("\\n🔬 Running advanced optimization...")
        best_configs = advanced_worker_tuning(dataset_path)
        
        if best_configs:
            best = best_configs[0]
            print(f"\\n✅ BEST CONFIGURATION:")
            print(f"   workers={best['workers']}, batch={best['batch']}, imgsz={best['imgsz']}")
    
    elif choice == "3":
        max_workers = int(input("Enter maximum workers to test: "))
        worker_list = list(range(0, max_workers + 1, 2))
        
        print(f"\\n🧪 Testing custom range: {worker_list}")
        results = benchmark_workers(worker_list)
        analyze_results(results)
    
    else:
        print("Invalid choice!")
        return
    
    print("\\n🎯 GENERAL RECOMMENDATIONS:")
    print("- Start with the optimal worker count found above")
    print("- Monitor CPU usage during training (should be < 90%)")
    print("- If CPU hits 100%, reduce workers by 2")
    print("- If training is still slow, check storage speed and GPU utilization")

if __name__ == "__main__":
    main()

# ================================
# QUICK REFERENCE GUIDE
# ================================

"""
🔧 QUICK WORKER GUIDELINES:

SYSTEM TYPE          | RECOMMENDED WORKERS
--------------------|--------------------
Low-end (4 cores)   | 2-4 workers
Mid-range (6-8 cores)| 4-6 workers  
High-end (10+ cores) | 6-8 workers
Server (16+ cores)   | 8-12 workers

⚠️  WARNING SIGNS:
- CPU usage > 95%: Reduce workers
- RAM usage > 90%: Reduce workers  
- Training slower with more workers: CPU bottleneck

✅ OPTIMAL INDICATORS:
- CPU usage 70-90%
- GPU utilization > 80%
- Stable RAM usage
- Fastest training time

🎯 TESTING STRATEGY:
1. Run quick test first
2. Use recommended workers for your dataset
3. Monitor system resources during actual training
4. Fine-tune based on performance
"""