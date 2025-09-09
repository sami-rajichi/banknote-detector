# YOLO Export Formats Guide 📦

A beginner-friendly guide to understanding YOLO model export formats, their trade-offs, and when to use each one.

## 🎯 Quick Summary Table

| Format | Speed | Size | Compatibility | Best For | Difficulty |
|--------|-------|------|---------------|----------|------------|
| **PyTorch** | 🐌 Slow | 🐘 Large | 🟡 Python only | Development | ⭐ Easy |
| **ONNX** | 🏃 Fast | 🎒 Medium | 🟢 Universal | Production | ⭐⭐ Easy |
| **TensorRT** | 🚀 Fastest | 🎒 Medium | 🔴 NVIDIA only | High-speed inference | ⭐⭐⭐ Hard |
| **CoreML** | 🏃 Fast | 🎒 Medium | 🔴 Apple only | iOS/macOS apps | ⭐⭐ Easy |
| **OpenVINO** | 🏃 Fast | 🎒 Medium | 🟡 Intel CPUs | Intel hardware | ⭐⭐⭐ Medium |
| **TensorFlow Lite** | 🏃 Fast | 🐁 Small | 🟢 Mobile/Edge | Mobile apps | ⭐⭐ Easy |
| **TorchScript** | 🐌 Medium | 🐘 Large | 🟡 PyTorch only | Production PyTorch | ⭐⭐ Easy |

---

## 📋 Detailed Format Breakdown

### 1. PyTorch (.pt) - The Original 🏠

**What it is:** The native PyTorch format - this is what YOLO saves by default.

#### ⚡ Performance
- **Speed**: Slowest (baseline reference)
- **Latency**: ~50-100ms per image
- **Memory**: High RAM usage
- **Size**: Largest files (100-500MB)

#### ✅ Pros
- No conversion needed - works immediately
- Full model functionality (training, inference, modifications)
- Best debugging capabilities
- Supports all YOLO features

#### ❌ Cons
- Requires Python + PyTorch installation
- Slowest inference speed
- Largest file size
- Not optimized for production

#### 🎯 Best Used For
- **Development and experimentation**
- **Model training and fine-tuning**
- **Research and debugging**
- **When you need full model flexibility**

#### 💡 ELI5 Explanation
*"Think of this like a Word document - it has all the features and you can edit everything, but it's big and slow to open."*

---

### 2. ONNX (.onnx) - The Universal Translator 🌍

**What it is:** An open standard that works with many different AI frameworks.

#### ⚡ Performance
- **Speed**: 2-3x faster than PyTorch
- **Latency**: ~20-40ms per image
- **Memory**: Medium RAM usage
- **Size**: Medium files (50-200MB)

#### ✅ Pros
- Works with multiple frameworks (PyTorch, TensorFlow, etc.)
- Good balance of speed and compatibility
- Widely supported by deployment tools
- Relatively easy to use

#### ❌ Cons
- Some features might not convert perfectly
- Requires ONNX runtime installation
- Larger than mobile-optimized formats

#### 🎯 Best Used For
- **Cross-platform deployment**
- **When you need good speed and compatibility**
- **Cloud inference services**
- **When switching between AI frameworks**

#### 💡 ELI5 Explanation
*"Like a PDF - it looks the same everywhere and most programs can open it, smaller than Word but you can't edit it easily."*

---

### 3. TensorRT (.engine) - The Speed Demon 🚀

**What it is:** NVIDIA's high-performance inference engine for their GPUs.

#### ⚡ Performance
- **Speed**: Fastest possible (5-10x faster than PyTorch)
- **Latency**: ~5-15ms per image
- **Memory**: Optimized GPU memory usage
- **Size**: Medium files (30-150MB)

#### ✅ Pros
- Extremely fast inference
- Optimized for specific GPU architecture
- Low latency for real-time applications
- Automatic mixed precision

#### ❌ Cons
- Only works on NVIDIA GPUs
- GPU-specific optimization (not portable)
- Complex setup and compilation
- Requires NVIDIA TensorRT installation

#### 🎯 Best Used For
- **Real-time applications (video processing, robotics)**
- **High-throughput inference servers**
- **When you have NVIDIA GPUs and need maximum speed**
- **Edge deployment on NVIDIA Jetson devices**

#### 💡 ELI5 Explanation
*"Like a race car engine - incredibly fast but only works in specific cars (NVIDIA GPUs) and needs expert mechanics to set up."*

---

### 4. CoreML (.mlpackage) - The Apple Specialist 🍎

**What it is:** Apple's machine learning framework for iOS, macOS, and Apple devices.

#### ⚡ Performance
- **Speed**: Fast on Apple devices
- **Latency**: ~15-30ms per image (on Apple silicon)
- **Memory**: Optimized for Apple hardware
- **Size**: Medium files (40-180MB)

#### ✅ Pros
- Optimized for Apple devices (iPhone, iPad, Mac)
- Good battery efficiency
- Integrates well with iOS/macOS apps
- Hardware acceleration on Apple Silicon

#### ❌ Cons
- Only works on Apple devices
- Limited to Apple's ecosystem
- Conversion can be tricky for complex models

#### 🎯 Best Used For
- **iOS/macOS applications**
- **iPhone/iPad apps with on-device inference**
- **Mac applications requiring ML**
- **When you need Apple ecosystem integration**

#### 💡 ELI5 Explanation
*"Like apps from the App Store - they work perfectly on iPhones and Macs, but you can't use them on Android or Windows."*

---

### 5. OpenVINO - The Intel Optimizer 🧠

**What it is:** Intel's toolkit for optimizing AI models on Intel hardware (CPUs, integrated GPUs).

#### ⚡ Performance
- **Speed**: Fast on Intel CPUs
- **Latency**: ~25-45ms per image
- **Memory**: Optimized CPU usage
- **Size**: Medium files (35-160MB)

#### ✅ Pros
- Optimized for Intel processors
- Good CPU performance
- Supports various Intel hardware
- Professional deployment tools

#### ❌ Cons
- Best only on Intel hardware
- Complex setup process
- Less common than other formats
- Requires OpenVINO runtime

#### 🎯 Best Used For
- **Intel CPU-based servers**
- **Edge devices with Intel processors**
- **When GPU is not available**
- **Enterprise Intel infrastructure**

#### 💡 ELI5 Explanation
*"Like software optimized for Intel computers - runs really well on Intel processors but doesn't help much on other brands."*

---

### 6. TensorFlow Lite (.tflite) - The Mobile Champion 📱

**What it is:** Google's lightweight format designed specifically for mobile and embedded devices.

#### ⚡ Performance
- **Speed**: Fast and efficient
- **Latency**: ~20-50ms per image
- **Memory**: Very low memory usage
- **Size**: Smallest files (10-80MB)

#### ✅ Pros
- Smallest file sizes
- Designed for mobile/embedded devices
- Low battery consumption
- Cross-platform mobile support
- Easy integration with mobile apps

#### ❌ Cons
- May have some accuracy loss due to optimization
- Limited functionality compared to full models
- Conversion process can be complex

#### 🎯 Best Used For
- **Mobile applications (Android/iOS)**
- **Embedded devices and IoT**
- **Edge computing with limited resources**
- **When file size and battery life matter most**

#### 💡 ELI5 Explanation
*"Like a compressed photo - much smaller file size and loads faster on your phone, but might lose a tiny bit of quality."*

---

### 7. TorchScript (.torchscript) - The Production PyTorch 🏭

**What it is:** PyTorch's format for production deployment without Python dependencies.

#### ⚡ Performance
- **Speed**: Faster than regular PyTorch
- **Latency**: ~30-60ms per image
- **Memory**: Medium RAM usage
- **Size**: Large files (90-450MB)

#### ✅ Pros
- No Python interpreter needed
- Faster than regular PyTorch models
- Still supports most PyTorch features
- Good for C++ applications

#### ❌ Cons
- Still tied to PyTorch ecosystem
- Larger files than optimized formats
- Not as fast as specialized formats

#### 🎯 Best Used For
- **Production PyTorch deployments**
- **C++ applications using PyTorch**
- **When you need PyTorch compatibility without Python**
- **Server deployments with PyTorch infrastructure**

#### 💡 ELI5 Explanation
*"Like a standalone app - you don't need to install extra software to run it, but it's still bigger than specialized formats."*

---

## 🤔 Which Format Should You Choose?

### For Beginners 👶
**Start with ONNX** - it's fast, widely supported, and works almost everywhere.

### For Mobile Apps 📱
**Use TensorFlow Lite** - smallest size and designed for phones.

### For Apple Ecosystem 🍎
**Use CoreML** - perfect integration with iOS/macOS.

### For Maximum Speed 🚀
**Use TensorRT** (if you have NVIDIA GPUs) - fastest possible inference.

### For Development 🛠️
**Stick with PyTorch** - full features and easy debugging.

---

## 📊 Real-World Performance Comparison

*Based on typical YOLO11m model on 640×640 images:*

| Scenario | Recommended Format | Expected Speed | File Size |
|----------|-------------------|----------------|-----------|
| **Mobile App** | TensorFlow Lite | 30-50ms | 20-40MB |
| **Web Service** | ONNX | 20-40ms | 80-150MB |
| **Real-time Video** | TensorRT | 5-15ms | 60-120MB |
| **iPhone App** | CoreML | 15-30ms | 70-140MB |
| **Development** | PyTorch | 50-100ms | 150-300MB |
| **Intel Server** | OpenVINO | 25-45ms | 70-130MB |

---

## 🚀 Quick Start Commands

```bash
# Export to ONNX (most common)
model = YOLO("best.pt")
model.export(format='onnx', imgsz=640)

# Export to TensorRT (NVIDIA GPUs)
model.export(format='engine', half=True)

# Export to mobile (TensorFlow Lite)
model.export(format='tflite', int8=True)

# Export to Apple devices
model.export(format='coreml')

# Export multiple formats at once
for fmt in ['onnx', 'tflite', 'coreml']:
    model.export(format=fmt, imgsz=640)
```

---

## ⚠️ Important Considerations

### Accuracy vs Speed Trade-off
- Faster formats might have slightly lower accuracy
- Test your specific use case to find the right balance
- Mobile formats prioritize size over precision

### Hardware Dependencies
- **TensorRT**: NVIDIA GPUs only
- **CoreML**: Apple devices only  
- **OpenVINO**: Best on Intel hardware
- **ONNX/TFLite**: Work everywhere

### Development Workflow
1. **Develop** with PyTorch (.pt)
2. **Test** with ONNX for compatibility
3. **Deploy** with format specific to your target platform

---

## 🔧 Troubleshooting Common Issues

### Export Fails
- Check YOLO version compatibility
- Ensure all dependencies are installed
- Try simpler model configurations first

### Slow Performance
- Verify you're using the right format for your hardware
- Check if GPU acceleration is working
- Consider model quantization (int8) for mobile

### Large File Sizes
- Use quantization (int8=True)
- Try TensorFlow Lite for smallest size
- Consider model pruning techniques

### Accuracy Loss
- Compare outputs between formats
- Avoid aggressive quantization if accuracy is critical
- Test with your specific dataset

---

Remember: **There's no "best" format** - only the best format for your specific use case! Start with ONNX for general use, then optimize based on your deployment requirements.