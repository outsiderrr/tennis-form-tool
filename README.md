# tennis-form-tool

网球动作可视化反馈工具 · A 阶（周末流水线）。设计蓝图见
`~/.gstack/projects/tennis-form-app/outsider-unknown-design-20260815-111019.md`。

v0 范围：**脚步分析**（背面机位、下半身指标）——骨骼叠加视频 + 膝角/站位宽度/重心时间线 + 指标 JSON。

## 环境

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python "mediapipe<1.0" opencv-python numpy matplotlib
mkdir -p models && curl -sL -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
```

注意：mediapipe 1.0.1 在 macOS 上初始化 Metal 会崩（DrishtiMetalHelper check failure），必须用 0.10.x。

## 用法

```bash
# 单视频分析（骨骼叠加 + 脚步指标 + 挥拍片段切割）
.venv/bin/python analyze.py <视频.MOV> [--start 秒] [--end 秒] [--cut-swings]

# 并排同步对比：两段挥拍按击球时刻对齐、按体型（小腿长）归一化
.venv/bin/python compare.py A.MOV B.MOV --ta 39.47 --tb 50.89 --labels "我的动作,参考动作"
```

compare.py 的击球时刻从 analyze.py 输出的 `swing_candidates` 里选；两侧可以是
同一视频的两次挥拍（自我对比），也可以一侧放参考/标准动作片段（同视角拍摄）。
帧率不同的两段视频（如 60fps vs 240fps）可直接对比——时间轴按秒对齐。

## 拍摄要求（血泪教训）

- **全身入画**，头顶留余量，人占画面高度 60–80%——缺头会让检测本身失败（实测有效帧从 ~95% 掉到 ~55%）
- 手机架到胸口以上高度
- 用 **240fps 慢动作**模式（60fps 常速在快速挥拍段有动作模糊）
- 背面机位测脚步；侧面机位测引拍/击球点（另拍）

## 已知局限（v0）

- 垫步检测受透视污染（像素坐标随人远近变化），待改为髋踝相对高度
- 挥拍检测依赖手腕可见度，上半身缺失时不可用
- 2D 投影量：跨机位角度对比会系统性失真，只做同视角对比
