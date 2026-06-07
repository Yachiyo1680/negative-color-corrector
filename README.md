# 底片自动校色 App — 技术框架

## 概述

从负片扫描图（已裁切）自动去除 C-41 橙色色罩，输出自然色彩的校色结果。

用户只需要：选正片/负片 → 手动裁切（可选）→ 点校色

---

## 系统架构

```
用户界面 (UI)
    │
    ▼
校色引擎 (Engine)
    ├─ 反相模块 (Invert)
    ├─ 色罩分析模块 (Mask Analyzer)
    ├─ 通道补偿模块 (Channel Compensator)
    ├─ 智能色阶模块 (Auto Levels)
    ├─ 暖调控制模块 (Warmth Controller)
    └─ AI偏色检测器 (Color Cast Detector) ← 反馈闭环
    │
    ▼
输出模块 (Exporter)
```

---

## 模块详解

### 1️⃣ 反相模块 — Invert

**输入：** 负片 RGB 图像（0-255）
**处理：** `pixel = 255 - pixel`
**输出：** 反相后的近似正像（但色罩还在）

仅负片模式下执行，正片模式直接跳过。

---

### 2️⃣ 色罩分析模块 — Mask Analyzer

**这是整个引擎的核心。**

#### 2.1 中性灰参考点采样

策略 A — 边缘片基采样（首选）：
```
取样区域：图片最左侧 5-10 列像素（未曝光片基）
计算该区域 R/G/B 均值 → mask_R, mask_G, mask_B
```

策略 B — 画面内中性灰检测（当边缘被裁掉时）：
```
在画面下半部分搜索亮度中等、饱和度最低的区域
候选区：左下栅栏、路面、阴影中的灰墙等
计算候选区 R/G/B 均值
```

策略 C — 如果 A 和 B 都不可靠，回退到全局灰世界假设：
```
取全图 R/G/B 均值作为参考
```

#### 2.2 补偿比例计算

```
// 对参考点在反相空间做中性化
inv_R = 255 - ref_R
inv_G = 255 - ref_G
inv_B = 255 - ref_B

target = (inv_R + inv_G + inv_B) / 3

scale_R = target / inv_R
scale_G = target / inv_G
scale_B = target / inv_B
```

---

### 3️⃣ 通道补偿模块 — Channel Compensator

**输入：** 反相后的图像 + 补偿比例
**处理：** 每通道乘以对应比例

```python
result_R = clamp(input_R × scale_R, 0, 255)
result_G = clamp(input_G × scale_G, 0, 255)
result_B = clamp(input_B × scale_B, 0, 255)
```

**输出：** 色罩基本消除的近似正像

---

### 4️⃣ 智能色阶模块 — Auto Levels

每通道独立做直方图拉伸：

```
对 R/G/B 各通道：
  lo = percentile(channel, 0.2%)    // 去掉极端暗点
  hi = percentile(channel, 99.8%)   // 去掉极端亮点
  new_channel = clamp((channel - lo) × 255 / (hi - lo), 0, 255)
```

这一步恢复对比度，避免色罩去除后画面发灰。

**参数建议：**
- 底线百分位：0.1% ~ 0.5%（可配置）
- 顶线百分位：99.5% ~ 99.9%（可配置）

---

### 5️⃣ 暖调控制模块 — Warmth Controller

色罩补偿后的画面通常会偏冷微蓝，需要加暖。

**关键规则：**
- ✅ 黄方向 = **R↑ + G↑**（红和绿一起加 → 黄色调，自然）
- ❌ 品红方向 = R↑ + B↑（红和蓝一起加 → 容易偏紫）

```python
// 默认暖调（偏自然）
warm_factor = 1.0 (用户可调 0.8~1.5)

result_R = clamp(input_R × (1.0 + 0.04 × warm_factor), 0, 255)
result_G = clamp(input_G × (1.0 + 0.02 × warm_factor), 0, 255)
result_B = clamp(input_B × (1.0 - 0.06 × warm_factor), 0, 255)
```

**可选风格预设：**
| 风格 | R | G | B |
|------|---|---|---|---|
| 自然 | +4% | +2% | -6% |
| Kodak Gold | +6% | +3% | -8% |
| Fuji Superia | +3% | +1% | -5% |
| 冷调（不推荐）| 0% | 0% | -2% |

---

### 6️⃣ AI 偏色检测器 — Color Cast Detector（反馈闭环）

**6.1 目的**
解决纯算法调色难以一次到位的问题——自动判断校色结果是否有偏色，然后回馈给补偿模块微调。

**6.2 工作流程**

```
校色引擎初跑 → AI检测偏色 → 判断是否OK？
  ├─ OK → 输出结果
  └─ 偏色 → 识别偏色方向（偏蓝/偏绿/偏品红/偏青）和程度
            → 微调通道补偿比例
            → 重跑校色引擎
            → 再次检测
```

**6.3 偏色检测指标**

AI 模型判断以下维度：
- 色温（过暖 / 过冷 / 适中）
- 色偏方向（绿 / 品红 / 蓝 / 黄）
- 中性灰置信度（画面中灰墙/路面是否真实中性）
- 整体自然度评分（0~1）

**6.4 微调策略**

| 检测结果 | 调整方向 |
|-----------|---------|
| 偏蓝 | R↑ +1%, B↓ -1% |
| 偏绿 | R↑ +1%, G↓ -1% |
| 偏品红 | G↑ +2%, R↓ -1% |
| 偏黄 | B↑ +2%, R↓ -1% |
| OK | 结束，输出 |

每次调整幅度不超过 3%，避免过冲。

---

### 7️⃣ 最终修饰

```python
// 可选
饱和度 = clamp(原饱和度 × 1.05~1.15)
对比度 = clamp(原对比度 × 1.03~1.08)
```

建议让用户可调节力度，或提供「自动优化」开关。

---

## 数据流总图

```
用户导入图片
    │
    ▼
┌──────────────────────────────────┐
│  手动操作区                       │
│  ├─ 选 负片/正片                  │
│  └─ 手动裁切（可选）              │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  校色引擎（自动）                 │
│                                  │
│  1. 反相（负片时）               │
│      │                          │
│  2. 色罩分析 ←─── 采样参考点     │
│      │                          │
│  3. 通道补偿                     │
│      │                          │
│  4. 智能色阶                     │
│      │                          │
│  5. 暖调控制                     │
│      │                          │
│  6. AI偏色检测 ──→ 有偏色 ──→ 3 │
│      │  (OK)                     │
│      ▼                          │
│  7. 最终修饰                     │
└──────┬───────────────────────────┘
       │
       ▼
    显示 + 导出
```

---

## 技术栈建议

| 层 | 推荐 |
|----|------|
| 图像处理 | Python + Pillow / OpenCV / NumPy |
| AI 偏色模型 | 轻量 CNN（MobileNet 量级）或直接复用 VL 模型判断 |
| GUI | 如果是桌面 App：PyQt / Tkinter 或 Electron |
| 如果是 Web 应用：Flask/FastAPI 后端 + 前端上传 |

---

## 边缘情况处理

| 情况 | 处理方式 |
|------|---------|
| 黑白负片 | 用户选正片模式，跳过色罩补偿（但可选反转） |
| 片基边缘被裁掉 | 切换到画面内中性灰检测（策略 B） |
| 画面里没有中性灰 | 回退到灰世界假设（策略 C）+ AI 反馈矫正 |
| 严重欠曝/过曝 | 智能色阶参数放宽，AI 检测优先 |
| 扫描偏色（V500 等自带色偏）| 两道补偿：先校扫描仪偏色，再做色罩补偿 |

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `core/engine.py` | 校色引擎主入口 |
| `core/invert.py` | 反相模块 |
| `core/mask_analyzer.py` | 色罩分析 + 补偿比例计算 |
| `core/channel_comp.py` | 通道补偿 |
| `core/auto_levels.py` | 智能色阶 |
| `core/warmth.py` | 暖调控制 |
| `core/cast_detector.py` | AI 偏色检测器 |
| `ui/` | 用户界面层 |
| `config.py` | 默认参数配置 |
| `main.py` | 启动入口 |

---

## 当前完成状态

| 模块 | 状态 |
|------|------|
| `core/invert.py` | ✅ 反相 |
| `core/mask_analyzer.py` | ✅ 色罩分析（三策略） |
| `core/channel_comp.py` | ✅ 通道补偿 |
| `core/auto_levels.py` | ✅ 智能色阶 |
| `core/warmth.py` | ✅ 暖调控制（4预设） |
| `core/cast_detector.py` | ✅ 启发式 + VL API + 工厂模式 |
| `core/engine.py` | ✅ 校色引擎（含反馈闭环） |
| `core/credential_store.py` | ✅ API Key 安全存储 |
| `core/config_manager.py` | ✅ JSON5 配置管理 |
| `core/model_provider.py` | ✅ 多 Provider 管理 |
| `main.py` | ✅ CLI 入口 |
| `ui/app.py` | ✅ Gradio Web UI |

## 快速开始

```bash
# 安装依赖
pip install numpy Pillow requests

# CLI 校色（负片）
python main.py scan.tiff

# CLI 校色（正片 + Kodak Gold 暖调）
python main.py scan.tiff --film-type positive --warmth kodak_gold

# Web UI（带 Gradio）
pip install gradio
python main.py --gui
```
