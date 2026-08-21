# tennis-form-tool

网球动作可视化反馈工具：手机视频 → 骨骼 → 脚步达标率 → 进度表 → 关键时刻卡。
面向自学的初学者，反馈原则是**可视、可验证、发生在你自己身体上**——先给测量值再给解释。

- **想让 AI 帮你用这个工具**：把 [PROMPT.md](PROMPT.md) 里的提示词发给你的 AI 会话（Claude Code / Codex / Gemini CLI 等）。
- **想了解项目为什么这么做、当前状态和口径**：读 [CONTEXT.md](CONTEXT.md)。

v0 范围：**脚步分析**（背面机位、下半身指标）——骨骼叠加视频 + 膝角/站位宽度/重心时间线 + 指标 JSON。

## 环境

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python "mediapipe<1.0" opencv-python numpy matplotlib
mkdir -p models && curl -sL -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
```

注意：mediapipe 1.0.1 在 macOS 上初始化 Metal 会崩（DrishtiMetalHelper check failure），必须用 0.10.x。

## 每周用法（一条命令）

```bash
# 拍完一段 → 全流程：姿态 → 达标率 → 与上一次并排的进度表 → 四维度示意视频
.venv/bin/python session.py <视频.MOV> --player me --segments "0-180:定点,180-360:变化" --label "8/22 背面"
# 结果在 sessions/<player>/<时间>_<视频名>/：progress.md 进度表，card.png 关键时刻卡，annot/ 示意视频
```

## 关键时刻卡（C 阶）

`card.png`：一次训练一张。横排四个关键时刻（① 分腿垫步 ② 引拍完成 ③ 击球瞬间 ④ 随挥收尾），
每格 = 从你自己的挥拍里自动截的一帧 + 火柴人 + 测量（●达标/○未达标）+ 一句外部焦点口令；
底部是四项在本次全部挥拍上的达标率（有上一次就带箭头）。展示的那一拍是本次「最接近达标」的一拍。
单独出卡：`card.py <视频> <landmarks.npz> --report report.json [--prev 上次report.json] --out card.png`

## 分步工具

```bash
# 单视频分析（骨骼叠加 + 脚步指标 + 挥拍片段切割）
.venv/bin/python analyze.py <视频.MOV> [--start 秒] [--end 秒] [--cut-swings] [--det-conf 0.5]
# 人物在画面里较小（<30% 高）时可降 --det-conf 到 0.3；再低会出现「幽灵骨架」误检，宁可重拍

# 侧面机位报告（真·侧面：镜头垂直于击球方向）：击球点在体前距离 / 真实膝角 / 弯腰 vs 屈膝
.venv/bin/python side.py <视频> <landmarks.npz> [--label ...]

# 全套脚步体检 / 多视频对照 / 段落分列
.venv/bin/python report.py out/A_landmarks.npz [out/B_landmarks.npz ...] [--segments 0-180:定点,180-360:变化]

# 单维度示意视频（站位/屈膝/垫步/击球后重心）
.venv/bin/python annotate.py <视频> <landmarks.npz> --t <击球秒> --all

# 两机位事后同步：靠两台手机都看见的同一串挥拍做互相关，不需要网络/时钟同步
.venv/bin/python sync.py A_landmarks.npz B_landmarks.npz     # 输出 B 相对 A 的偏移秒数与置信度
.venv/bin/python sync.py --selftest out/X_landmarks.npz     # 自测（人为错开 3.37s，应恢复到一帧内）

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
- 背面机位测脚步；侧面机位测引拍/击球点（另拍或双机同拍都行）
- **侧面机位必须垂直于击球方向**（正手朝发球机打 → 相机在你右侧腰胸高、6–8 米、人占画面 ≥50%）。放在发球机方向 = 正面机位，侧面指标全部失效
- 第二机位宁可近勿远：人 <30% 画面高会跌破检测门槛；降低 --det-conf 会引入背景误检
- 双机位不需要同时按开始：两台各自开录整段不暂停，事后 `sync.py` 自动对齐；开拍时在两台都能看见的位置举手拍一下，峰更尖

## 已知局限

- 2D 投影量：背面机位会低估屈膝幅度；跨机位角度不可直接比，只做同视角对比
- 挥拍定位靠手腕速度峰 + 视觉规则剔假（身份/深蹲/捡球）；音频击球声方案已测试不可靠，弃用
- 多人场景：目标锁定按连续性 + 身高，人物长时间被裁出画面时可能留空（宁可留空不锁错）
- 阈值（150°、1.3–1.8、0.10 等）为原理占位口径，看趋势不抠个位数；正式口径待与教练/文献统一
- 身份约束假设「手持跟拍、目标在画面里大小基本稳定」：候选身高 < 0.65×长期中位一律不接受（宁可留空）
- 卡片里的「转肩(粗估)」= 肩线视宽/髋线视宽，背面机位不可靠，只展示不计入达标；上半身正式指标待侧面机位
