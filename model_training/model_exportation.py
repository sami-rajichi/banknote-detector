#!/usr/bin/env python3
"""
YOLO Model Export Script
Exports trained YOLO models to various formats for different deployment scenarios
"""

import os
import time
import torch
from pathlib import Path
from ultralytics import YOLO
import psutil
import json

def create_export_directory(base_path="./exported_models"):
    """Create organized directory structure for exports"""
    export_dir = Path(base_path)
    export_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each format
    formats = [
        # 'pytorch', 
        'onnx', 
        'tensorrt', 
        # 'coreml', 
        # 'openvino', 
        'tflite', 
        # 'torchscript'
    ]
    
    for fmt in formats:
        (export_dir / fmt).mkdir(exist_ok=True)
    
    return export_dir

def get_file_size_mb(filepath):
    """Get file size in MB"""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0

def measure_export_time(export_func):
    """Decorator to measure export time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = export_func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

@measure_export_time
def export_pytorch(model, export_dir, model_name):
    """Export to PyTorch format (native)"""
    output_path = export_dir / "pytorch" / f"{model_name}.pt"
    # PyTorch models are already in .pt format, just copy
    import shutil
    shutil.copy(model.ckpt_path, output_path)
    return str(output_path)

@measure_export_time
def export_onnx(model, export_dir, model_name, dynamic=False, simplify=True):
    """Export to ONNX format"""
    output_path = export_dir / "onnx" / f"{model_name}.onnx"
    exported = model.export(
        format='onnx',
        dynamic=dynamic,
        simplify=simplify,
        workspace=4,  # GB
        imgsz=640
    )
    
    # Move to organized directory
    import shutil
    if os.path.exists(exported):
        shutil.move(exported, output_path)
    
    return str(output_path)

@measure_export_time
def export_tensorrt(model, export_dir, model_name, half=True):
    """Export to TensorRT format (NVIDIA GPUs only)"""
    if not torch.cuda.is_available():
        print("⚠️  TensorRT export requires CUDA GPU")
        return None, 0
    
    try:
        output_path = export_dir / "tensorrt" / f"{model_name}.engine"
        exported = model.export(
            format='engine',
            half=half,
            imgsz=640
        )
        
        # Move to organized directory
        import shutil
        if os.path.exists(exported):
            shutil.move(exported, output_path)
        
        return str(output_path)
    except Exception as e:
        print(f"❌ TensorRT export failed: {e}")
        return None, 0

@measure_export_time
def export_coreml(model, export_dir, model_name, int8=False, half=False):
    """Export to CoreML format (Apple devices)"""
    try:
        output_path = export_dir / "coreml" / f"{model_name}.mlpackage"
        exported = model.export(
            format='coreml',
            int8=int8,
            half=half,
            imgsz=640
        )
        
        # Move to organized directory
        import shutil
        if os.path.exists(exported):
            shutil.move(exported, output_path)
        
        return str(output_path)
    except Exception as e:
        print(f"❌ CoreML export failed: {e}")
        return None, 0

@measure_export_time
def export_openvino(model, export_dir, model_name, half=False):
    """Export to OpenVINO format (Intel optimization)"""
    try:
        output_dir = export_dir / "openvino" / model_name
        output_dir.mkdir(exist_ok=True)
        
        exported = model.export(
            format='openvino',
            half=half,
            imgsz=640
        )
        
        # OpenVINO creates multiple files, move the directory
        import shutil
        if os.path.exists(exported):
            # Move contents to organized directory
            for item in os.listdir(os.path.dirname(exported)):
                src = os.path.join(os.path.dirname(exported), item)
                dst = os.path.join(output_dir, item)
                if os.path.isfile(src):
                    shutil.move(src, dst)
        
        return str(output_dir)
    except Exception as e:
        print(f"❌ OpenVINO export failed: {e}")
        return None, 0

@measure_export_time
def export_tflite(model, export_dir, model_name, int8=False):
    """Export to TensorFlow Lite format"""
    try:
        output_path = export_dir / "tflite" / f"{model_name}.tflite"
        exported = model.export(
            format='tflite',
            int8=int8,
            imgsz=640
        )
        
        # Move to organized directory
        import shutil
        if os.path.exists(exported):
            shutil.move(exported, output_path)
        
        return str(output_path)
    except Exception as e:
        print(f"❌ TFLite export failed: {e}")
        return None, 0

@measure_export_time
def export_torchscript(model, export_dir, model_name, optimize=True):
    """Export to TorchScript format"""
    try:
        output_path = export_dir / "torchscript" / f"{model_name}.torchscript"
        exported = model.export(
            format='torchscript',
            optimize=optimize,
            imgsz=640
        )
        
        # Move to organized directory
        import shutil
        if os.path.exists(exported):
            shutil.move(exported, output_path)
        
        return str(output_path)
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
        return None, 0

def benchmark_inference(model_path, format_type, num_runs=10):
    """Simple inference benchmark"""
    try:
        if format_type == 'pytorch':
            model = YOLO(model_path)
        elif format_type == 'onnx':
            model = YOLO(model_path, task='detect')
        else:
            # For other formats, try loading with YOLO
            model = YOLO(model_path)
        
        # Warm up
        dummy_img = torch.randn(1, 3, 640, 640)
        for _ in range(3):
            try:
                model.predict(dummy_img, verbose=False)
            except:
                pass
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            try:
                model.predict(dummy_img, verbose=False)
                times.append((time.time() - start_time) * 1000)  # Convert to ms
            except:
                return None
        
        return {
            'avg_latency_ms': sum(times) / len(times),
            'min_latency_ms': min(times),
            'max_latency_ms': max(times)
        }
    except Exception as e:
        print(f"Benchmark failed for {format_type}: {e}")
        return None

def export_all_formats(model_paths, export_base_dir="./exported_models"):
    """
    Export multiple YOLO models to all supported formats
    
    Args:
        model_paths (dict): Dictionary with model names and paths
        export_base_dir (str): Base directory for exports
    """
    
    results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n🔄 Exporting {model_name}...")
        print("=" * 50)
        
        # Create model-specific export directory
        model_export_dir = create_export_directory(f"{export_base_dir}/{model_name}")
        
        # Load model
        try:
            model = YOLO(model_path)
            print(f"✅ Loaded model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model {model_path}: {e}")
            continue
        
        model_results = {
            'source_path': model_path,
            'source_size_mb': get_file_size_mb(model_path),
            'exports': {}
        }
        
        # Export to different formats
        export_functions = [
            # ('pytorch', export_pytorch),
            ('onnx', export_onnx),
            ('tensorrt', export_tensorrt),
            # ('coreml', export_coreml),
            # ('openvino', export_openvino),
            # ('tflite', export_tflite),
            # ('torchscript', export_torchscript)
        ]
        
        for format_name, export_func in export_functions:
            print(f"\n📦 Exporting to {format_name.upper()}...")
            
            try:
                exported_path, export_time = export_func(model, model_export_dir, model_name)
                
                if exported_path:
                    file_size = get_file_size_mb(exported_path)
                    
                    # Quick benchmark
                    print(f"🔍 Benchmarking {format_name}...")
                    benchmark_results = benchmark_inference(exported_path, format_name)
                    
                    model_results['exports'][format_name] = {
                        'path': exported_path,
                        'size_mb': file_size,
                        'export_time_seconds': round(export_time, 2),
                        'benchmark': benchmark_results,
                        'success': True
                    }
                    
                    print(f"✅ {format_name}: {file_size:.1f} MB, {export_time:.1f}s export time")
                    if benchmark_results:
                        print(f"   ⚡ Avg inference: {benchmark_results['avg_latency_ms']:.1f}ms")
                
                else:
                    model_results['exports'][format_name] = {
                        'success': False,
                        'error': 'Export failed'
                    }
                    print(f"❌ {format_name}: Export failed")
                    
            except Exception as e:
                model_results['exports'][format_name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ {format_name}: {str(e)}")
        
        results[model_name] = model_results
    
    # Save results to JSON
    results_path = f"{export_base_dir}/export_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Export results saved to: {results_path}")
    return results

def print_summary(results):
    """Print a summary of export results"""
    print("\n" + "="*60)
    print("📊 EXPORT SUMMARY")
    print("="*60)
    
    for model_name, model_data in results.items():
        print(f"\n🎯 {model_name.upper()}")
        print(f"   Source: {model_data['source_size_mb']:.1f} MB")
        
        successful_exports = []
        failed_exports = []
        
        for format_name, export_data in model_data['exports'].items():
            if export_data['success']:
                successful_exports.append((format_name, export_data))
            else:
                failed_exports.append(format_name)
        
        print(f"   ✅ Successful: {len(successful_exports)}")
        print(f"   ❌ Failed: {len(failed_exports)}")
        
        if successful_exports:
            print("\n   📦 Export Details:")
            for fmt, data in successful_exports:
                size_mb = data.get('size_mb', 0)
                benchmark = data.get('benchmark')
                latency = f"{benchmark['avg_latency_ms']:.1f}ms" if benchmark else "N/A"
                print(f"      {fmt:12}: {size_mb:6.1f} MB | {latency:8} | {data['path']}")

def main():
    """Main export function"""
    
    # Define your trained models here
    model_paths = {
        "yolo11x_banknote": "./model_outputs/banknote_detection_yolo11x_outputs/weights/best.pt", 
    }
    
    print("🚀 YOLO Model Export Tool")
    print("="*50)
    print("This will export your models to multiple formats:")
    # print("• PyTorch (.pt) - Native format")
    print("• ONNX (.onnx) - Cross-platform")
    print("• TensorRT (.engine) - NVIDIA GPU optimization") 
    # print("• CoreML (.mlpackage) - Apple devices")
    # print("• OpenVINO - Intel optimization")
    print("• TensorFlow Lite (.tflite) - Mobile/embedded")
    # print("• TorchScript (.torchscript) - Production PyTorch")
    
    # Check system info
    print(f"\n💻 System Info:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CPU Cores: {psutil.cpu_count()}")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Filter existing models
    existing_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            existing_models[name] = path
            print(f"✅ Found: {name} -> {path}")
        else:
            print(f"❌ Missing: {name} -> {path}")
    
    if not existing_models:
        print("\n❌ No model files found! Please check your paths.")
        return
    
    # Confirm export
    response = input(f"\n🤔 Export {len(existing_models)} models? This may take several minutes. (y/n): ")
    if response.lower() != 'y':
        print("Export cancelled.")
        return
    
    # Run exports
    results = export_all_formats(existing_models)
    
    # Print summary
    print_summary(results)
    
    print(f"\n🎉 Export completed! Check the './exported_models' directory.")
    print(f"📄 Detailed results saved in: './exported_models/export_results.json'")

if __name__ == "__main__":
    main()

# ================================
# USAGE EXAMPLES:
# ================================

"""
1. Basic Usage:
   python export_models.py

2. Custom Export (in your code):
   from export_models import export_all_formats
   
   models = {
       "my_model": "./path/to/model.pt"
   }
   results = export_all_formats(models, "./my_exports")

3. Single Format Export:
   model = YOLO("model.pt")
   model.export(format='onnx', imgsz=640, optimize=True)

4. Check Exported Model:
   # Load ONNX model
   model_onnx = YOLO("exported_models/yolo11x/onnx/yolo11x_banknote.onnx")
   results = model_onnx.predict("test_image.jpg")
"""