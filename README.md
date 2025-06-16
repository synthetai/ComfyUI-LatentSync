# LatentSync ComfyUI 节点

🎬 **专为ComfyUI设计的高级口型同步节点**

独立部署包，支持智能混合人脸处理，无需完整LatentSync项目。

## 🚀 快速开始

```bash
# 1. 复制节点到ComfyUI
cp -r comfyui_latentsync /path/to/ComfyUI/custom_nodes/LatentSync

# 2. 安装依赖
cd /path/to/ComfyUI && pip install -r custom_nodes/LatentSync/requirements.txt

# 3. 下载模型（在LatentSync项目目录下）
cd /path/to/LatentSync
huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints

# 4. 重启ComfyUI，搜索"LatentSync"节点开始使用！
```

## ✨ 特色功能

- 🎯 **智能混合人脸处理**：自动处理有/无人脸的帧，消除"Face not detected"错误
- 🔧 **多种人脸检测模式**：lenient/default/very_lenient/custom四种检测模式
- ⚡ **性能优化**：DeepCache加速和批处理优化
- 🎨 **ComfyUI原生支持**：完全符合ComfyUI节点规范
- 📦 **独立部署**：无需完整LatentSync项目

## 📦 安装方法

### 步骤1：复制节点文件
```bash
# 复制到ComfyUI自定义节点目录
cp -r comfyui_latentsync /path/to/ComfyUI/custom_nodes/LatentSync
```

### 步骤2：安装Python依赖
```bash
# 进入ComfyUI目录
cd /path/to/ComfyUI

# 激活ComfyUI环境（如果使用虚拟环境）
# conda activate comfyui  # 或其他环境激活命令

# 安装依赖
pip install -r custom_nodes/LatentSync/requirements.txt
```

### 步骤3：安装huggingface-cli（如果没有）
```bash
pip install huggingface_hub
```

## 📥 模型下载

### 步骤4：下载模型文件
```bash
# 在LatentSync项目根目录下执行（不是ComfyUI目录）
cd /path/to/LatentSync

# 下载模型文件
huggingface-cli download ByteDance/LatentSync-1.5 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt --local-dir checkpoints
```

模型将下载到 `LatentSync/checkpoints/` 目录，节点会自动识别和使用。

### 目录结构说明
安装完成后的目录结构应该是：
```
LatentSync/                    # 原项目目录
├── checkpoints/               # 模型文件目录
│   ├── latentsync_unet.pt
│   ├── whisper/tiny.pt
│   └── auxiliary/
└── ...

ComfyUI/                       # ComfyUI目录
└── custom_nodes/
    └── LatentSync/            # 节点安装位置
        ├── config.json        # 配置文件（自动生成）
        ├── models/            # 内嵌模型代码
        ├── pipelines/         # 内嵌管道代码
        └── ...
```

## ⚙️ 配置说明

首次运行时会自动创建配置文件 `config.json`，默认使用以下路径：

```json
{
  "model_paths": {
    "unet_checkpoint": "../checkpoints/latentsync_unet.pt",
    "whisper_model": "../checkpoints/whisper/tiny.pt",
    "auxiliary_models": "../checkpoints/auxiliary",
    "vae_model": "stabilityai/sd-vae-ft-mse"
  }
}
```

## 🎮 使用方法

### 步骤5：启动和使用
1. **重启ComfyUI**
2. **添加节点**：在节点浏览器中搜索 "LatentSync Lip Sync (Standalone)"
3. **连接输入**：
   - `video_path`：输入视频文件路径
   - `audio_path`：目标音频文件路径
4. **配置参数**：
   - `inference_steps`：推理步数（默认20）
   - `guidance_scale`：引导系数（默认2.0）
   - `face_detection_mode`：人脸检测模式（推荐lenient）
5. **执行工作流**

## 🔧 人脸检测模式

| 模式 | 适用场景 | 检测精度 | 成功率 |
|------|---------|---------|--------|
| **lenient** | 大多数视频（推荐） | 宽松 | ~90% |
| **default** | 高质量视频 | 标准 | ~75% |
| **very_lenient** | 困难视频，远距离人脸 | 非常宽松 | ~95% |
| **custom** | 自定义参数 | 可调节 | 可变 |

## 🚀 性能建议

### 速度优化
- 使用 `inference_steps`: 15-20
- 选择 `lenient` 人脸检测模式
- 处理较短的视频片段

### 质量优化
- 增加 `inference_steps`: 25-30
- 使用 `guidance_scale`: 2.5-3.0
- 确保输入视频质量良好

## 🔍 常见问题

### Q: "模型文件未找到"错误
**解决方案**：
1. 确认已运行模型下载命令
2. 检查 `checkpoints/` 目录是否存在模型文件
3. 如果需要，手动编辑 `config.json` 中的路径

### Q: "Face not detected"错误
**解决方案**：
1. 尝试 `very_lenient` 人脸检测模式
2. 确认视频包含清晰可见的人脸
3. 检查视频质量和光照条件

### Q: "内存不足"错误
**解决方案**：
1. 减少 `inference_steps` 到15
2. 降低 `guidance_scale` 到1.5
3. 处理较短的视频片段

### Q: 依赖安装失败
**解决方案**：
```bash
# 手动安装关键依赖
pip install torch diffusers transformers mediapipe insightface
pip install decord librosa DeepCache
```



## 🆘 获取帮助

遇到问题时：
1. 查看ComfyUI控制台输出的详细错误信息
2. 确认所有模型文件已正确下载
3. 尝试不同的人脸检测模式
4. 检查输入视频和音频文件格式

## 📄 许可证

Apache License 2.0

## 🙏 致谢

- 原版LatentSync：ByteDance
- ComfyUI集成：synthetai
- 增强功能：混合人脸处理，独立打包

---

**�� 开箱即用的口型同步解决方案！** 