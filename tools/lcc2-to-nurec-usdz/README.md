# LCC2 转 NuRec USDZ 工具

本工具用于将 **XGRIDS LCC2 streamed SOG 场景**转换成 Isaac Sim / Isaac Lab / Omniverse Kit 可打开的 **NVIDIA 3DGRUT NuRec USDZ 资产**。

它主要解决以下问题：

* `.lcc2` 不是普通点云文件，而是一个带 LOD 的流式场景索引。
* `data/3dgs/*.sog` 不能直接全部拼接，否则会把多层 LOD 重复叠加，导致重影、模糊、发虚。
* SOG V2 的 `means` 坐标是 log-space 压缩坐标，解码顺序错误会导致完整场景被压缩成一个很小的区域。
* Isaac Sim / Isaac Lab / Omniverse Kit 不能直接渲染 `.sog`，需要转换成 3DGRUT NuRec USDZ。

最终推荐在 Isaac Lab 中打开：

```text
*_nurec_with_collision.usdz
```

该文件包含：

* NuRec `UsdVol.Volume`，用于高斯渲染；
* 内嵌 `.nurec` 高斯 payload；
* 合并后的 invisible mesh，用作 collision / proxy 几何。

---

## 1. 输入目录结构

本工具面向常见的 LCC2 导出结构：

```text
lcc2-result/
├── 场景名.lcc2
├── data/
│   ├── 3dgs/
│   │   ├── 0_0.sog
│   │   ├── ...
│   │   └── env.sog
│   └── mesh/
│       ├── 0_0_0.ply
│       ├── ...
│       └── *.btree
└── info/
    ├── poses.json
    └── report.json
```

其中最重要的是：

```text
.lcc2
```

记录场景索引、LOD 树、SOG 文件范围、mesh 文件列表。

```text
data/3dgs/*.sog
```

高斯 splat 数据，通常由多个 SOG chunk 组成。

```text
data/mesh/*.ply
```

mesh 瓦片，可合并后作为 Isaac collision / proxy mesh。

---

## 2. 输出文件结构

给定一个 LCC2 场景目录，工具会输出：

```text
<output>/
├── <scene>_lod<depth>_gaussians.ply
├── <scene>_mesh.ply
├── <scene>_lod<depth>_nurec.usdz
├── <scene>_lod<depth>_nurec_with_collision.usdz
└── <scene>_summary.json
```

各文件含义如下。

### 2.1 `*_gaussians.ply`

标准 3DGS PLY 文件，包含：

```text
x y z
SH DC color
opacity logit
log-scale
quaternion rotation
```

高阶 SH 当前写成 0 填充字段，用于兼容 3DGRUT PLY importer。

### 2.2 `*_mesh.ply`

将 `data/mesh/*.ply` 合并后的 binary little-endian triangle mesh。

它主要用于 Isaac 中的 collision / proxy 几何。

### 2.3 `*_nurec.usdz`

只包含高斯 NuRec Volume 的 USDZ。

### 2.4 `*_nurec_with_collision.usdz`

高斯 NuRec Volume + invisible collision mesh。

Isaac Lab 中推荐优先使用这个文件。

### 2.5 `*_summary.json`

记录完整转换信息，包括：

* 输入路径；
* manifest 路径；
* 选中的 LOD depth；
* Gaussian 数量；
* Gaussian bbox；
* 输出文件路径；
* NuRec USDZ 验证结果；
* mesh 统计信息。

---

## 3. 安装方式

本工具分成两层依赖：

1. **基础转换依赖**：用于读取 LCC2 / SOG，并导出 3DGS PLY 和 mesh PLY。
2. **NuRec USDZ 后端依赖**：用于调用 NVIDIA 3DGRUT，将 PLY 转成 NuRec USDZ。

如果你只是想检查 SOG 解码是否正常，只需要安装基础依赖。

如果你想在 Isaac Sim / Isaac Lab / Omniverse Kit 中打开场景，则还需要安装 3DGRUT 后端。

---

## 3.1 基础安装：只导出 PLY

进入当前仓库：

```bash
cd /workspace/data/repos/unitree_rl_lab
```

安装基础依赖：

```bash
python3 -m pip install -r tools/lcc2-to-nurec-usdz/requirements.txt
```

基础依赖主要包括：

```text
numpy
pillow
```

安装完成后，测试工具是否可用：

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py --help
```

如果能正常打印参数说明，说明基础环境可用。

---

## 3.2 只导出 PLY 进行调试

如果暂时不想安装 3DGRUT，可以先只导出 PLY：

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root "/workspace/data/lcc2_sog_src/3D模型/lcc2-result" \
  --output-dir /workspace/data/lcc2_from_sog_tool \
  --lod-depth highest \
  --skip-usdz \
  --force
```

成功后，输出目录中应该能看到：

```text
*_gaussians.ply
*_mesh.ply
*_summary.json
```

可以查看 summary：

```bash
cat /workspace/data/lcc2_from_sog_tool/*_summary.json
```

重点检查：

```text
selected_depth
selected_gaussians
gaussian_bbox_min
gaussian_bbox_max
```

如果 `gaussian_bbox_min/max` 范围很小，而 mesh 范围很大，通常说明 SOG V2 means 解码顺序有问题。

---

## 3.3 完整安装：导出 NuRec USDZ

如果需要生成 Isaac / Omniverse 可打开的 NuRec USDZ，需要单独安装 NVIDIA 3DGRUT。

不建议将 3DGRUT 写入本工具的普通 `requirements.txt`，因为它依赖：

```text
PyTorch
CUDA
USD
Slang
3DGRUT
```

这些依赖和显卡驱动、CUDA 版本强相关，应该放在独立环境中管理。

一个已经验证过的安装方式如下。

首先安装 `uv`：

```bash
python3 -m pip install uv
```

克隆 3DGRUT：

```bash
git clone https://github.com/nv-tlabs/3dgrut.git /tmp/3dgrut_src
cd /tmp/3dgrut_src
```

创建 Python 3.12 虚拟环境：

```bash
uv venv .venv --python 3.12
```

安装 3DGRUT：

```bash
UV_INDEX='pytorch=https://download.pytorch.org/whl/cu128' \
  uv pip install -e . --index-strategy unsafe-best-match
```

安装 Slang 编译器：

```bash
UV_PROJECT_ENVIRONMENT=/tmp/3dgrut_src/.venv \
  bash scripts/install_slangc.sh
```

测试 3DGRUT 是否安装成功：

```bash
/tmp/3dgrut_src/.venv/bin/python -c "import threedgrut; print('3DGRUT OK')"
```

如果输出：

```text
3DGRUT OK
```

说明 3DGRUT 后端环境可用。

---

## 3.4 已验证的 3DGRUT 后端环境

开发时验证过的后端环境如下：

```text
threedgrut 1.1.0
torch 2.11.0+cu128
torchvision 0.26.0+cu128
usd-core 26.5
slangc 2026.5.2
```

对应说明文件：

```text
tools/lcc2-to-nurec-usdz/requirements-3dgrut-backend.txt
```

注意：

```text
requirements-3dgrut-backend.txt 只是后端环境说明，不建议直接 pip install。
```

---

## 4. 一键完整转换

进入当前仓库：

```bash
cd /workspace/data/repos/unitree_rl_lab
```

运行完整转换：

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root "/workspace/data/lcc2_sog_src/3D模型/lcc2-result" \
  --output-dir /workspace/data/lcc2_from_sog_tool \
  --lod-depth highest \
  --threedgrut-python /tmp/3dgrut_src/.venv/bin/python \
  --collision \
  --force
```

参数说明：

```text
--input-root
```

LCC2 场景目录，里面应该包含 `.lcc2` 和 `data/`。

```text
--output-dir
```

输出目录。

```text
--lod-depth highest
```

只导出最高精度 LOD，避免把多层 LOD 重复拼接。

```text
--threedgrut-python
```

指定安装了 `threedgrut` 的 Python。

```text
--collision
```

将合并后的 mesh 写入 USDZ，作为 invisible collision / proxy mesh。

```text
--force
```

覆盖已有输出文件。

---

## 5. 在 Isaac Lab / Isaac Sim 中打开

转换成功后，优先打开：

```text
*_nurec_with_collision.usdz
```

当前项目中已验证的输出文件为：

```text
/workspace/data/lcc2_from_sog_tool/444ab7c3911871d73a84e531866561ff_lod4_nurec_with_collision.usdz
```

可以使用项目脚本打开：

```bash
bash scripts/_internal/entry/open_substation_usdz_gui.sh \
  /workspace/data/lcc2_from_sog_tool/444ab7c3911871d73a84e531866561ff_lod4_nurec_with_collision.usdz
```

如果不确定真实文件名，可以查看 summary：

```bash
cat /workspace/data/lcc2_from_sog_tool/*_summary.json
```

重点查看：

```json
"nurec_usdz": "...",
"collision_usdz": "..."
```

Isaac Lab 中推荐使用：

```text
collision_usdz
```

---

## 6. 常用命令

### 6.1 查看帮助

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py --help
```

### 6.2 只导出 PLY

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root /path/to/lcc2-result \
  --output-dir /tmp/lcc2-debug \
  --lod-depth highest \
  --skip-usdz \
  --force
```

### 6.3 完整导出 NuRec USDZ

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root /path/to/lcc2-result \
  --output-dir /tmp/lcc2-nurec \
  --lod-depth highest \
  --threedgrut-python /tmp/3dgrut_src/.venv/bin/python \
  --collision \
  --force
```

### 6.4 不合并 mesh

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root /path/to/lcc2-result \
  --output-dir /tmp/lcc2-nurec \
  --lod-depth highest \
  --threedgrut-python /tmp/3dgrut_src/.venv/bin/python \
  --skip-mesh \
  --force
```

### 6.5 包含 env.sog

通常不需要包含 `env.sog`。

如果确实需要，可以使用：

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root /path/to/lcc2-result \
  --output-dir /tmp/lcc2-nurec \
  --lod-depth highest \
  --include-env \
  --force
```

---

## 7. 参数说明

### 必需参数

```text
--input-root
```

包含 `.lcc2` 和 `data/` 的 LCC2 场景目录。

```text
--output-dir
```

输出目录。

### 常用可选参数

```text
--manifest
```

手动指定 `.lcc2` 文件。当目录里有多个 `.lcc2` 时使用。

```text
--lod-depth
```

选择 LOD 层，可以是：

```text
highest
lowest
整数 depth
```

默认推荐：

```text
--lod-depth highest
```

```text
--collision
```

将合并 mesh 加入 USDZ，作为 invisible collision / proxy mesh。

```text
--threedgrut-python
```

指定安装了 `threedgrut` 的 Python。

例如：

```text
/tmp/3dgrut_src/.venv/bin/python
```

```text
--skip-usdz
```

只生成 PLY，不调用 3DGRUT。

```text
--skip-mesh
```

不合并 mesh。

```text
--include-env
```

包含 `env.sog`。通常不需要。

```text
--force
```

覆盖已有输出文件。

```text
--summary
```

手动指定 summary JSON 输出路径。

---

## 8. 验证标准

一个正常的 summary 应该类似：

```json
{
  "selected_depth": 4,
  "selected_gaussians": 2416905,
  "gaussian_bbox_min": [-24.0488, -22.1797, -6.2266],
  "gaussian_bbox_max": [11.2279, 13.3296, 5.6443],
  "validation": {
    "collision_usdz": {
      "has_nurec_payload": true,
      "has_nurec_volume_marker": true,
      "has_mesh_usd": true
    }
  }
}
```

请重点检查：

* `selected_depth` 是否为最高精度 LOD depth；
* `selected_gaussians` 是否等于最高精度 LOD 数量；
* `gaussian_bbox_min/max` 是否接近 `.lcc2` root bbox；
* `has_nurec_payload` 是否为 `true`；
* `has_nurec_volume_marker` 是否为 `true`；
* 如果使用了 `--collision`，`has_mesh_usd` 是否为 `true`。

---

## 9. 为什么不能直接拼接所有 SOG？

LCC2 是 streamed LOD 格式。

`.lcc2` 里面记录了每个 LOD 层对应的 `name/start/count` 范围。

例如一个场景可能是：

```text
lodSplats: [2416905, 1208833, 604447, 302188]

tree depth 1: 302188
tree depth 2: 604447
tree depth 3: 1208833
tree depth 4: 2416905
```

这些数字加起来不是一个“更完整的场景”，而是同一场景的多层 LOD 表示。

如果把所有 SOG 文件、所有 LOD 层全部 concat 到一个 PLY：

* 同一物体会被多层精度重复显示；
* Isaac 里会出现重影、模糊、发虚；
* USDZ 文件会变得很大；
* 渲染质量反而下降。

因此本工具默认推荐使用：

```bash
--lod-depth highest
```

也就是只导出最深层、最高精度的 LOD。

---

## 10. SOG V2 坐标解码重点

SOG V2 的 `means` 不是普通线性坐标，而是 log-space 压缩坐标。

正确解码顺序是：

```text
uint16
  -> 在 meta.means.mins / maxs 之间线性插值
  -> inverse log transform
```

也就是：

```text
先映射到 log-space，再做 invLogTransform
```

不能反过来。

错误顺序会导致完整场景被压缩成一个很小的局部区域，看起来像：

```text
高斯只显示一个小房间，但 mesh 却很大。
```

本工具已经按照官方 SOG reader 的实现修正了解码顺序。

---

## 11. 常见问题

### 11.1 Isaac 里有重影、模糊、发虚

通常原因是把所有 LOD 层都拼起来了。

解决方式：

```bash
--lod-depth highest
```

并检查 summary 中的：

```text
lod_counts_by_depth
selected_depth
selected_gaussians
```

---

### 11.2 高斯只显示一个小房间，mesh 却很大

通常原因是 SOG V2 `means` 解码顺序错误。

正确顺序必须是：

```text
uint16 -> meta.means.mins/maxs 插值 -> inverse log transform
```

本工具已经按官方 SOG reader 的实现修正。

如果仍然出现该问题，请优先检查 summary 中的：

```text
gaussian_bbox_min
gaussian_bbox_max
```

并与 `.lcc2` root bbox 对比。

---

### 11.3 `ModuleNotFoundError: threedgrut`

说明当前 Python 不是 3DGRUT 环境。

错误示例：

```text
ModuleNotFoundError: No module named 'threedgrut'
```

解决方式：

```bash
--threedgrut-python /tmp/3dgrut_src/.venv/bin/python
```

同时可以先测试：

```bash
/tmp/3dgrut_src/.venv/bin/python -c "import threedgrut; print('3DGRUT OK')"
```

---

### 11.4 mesh 挡住了高斯

请使用：

```bash
--collision
```

生成的 mesh 会被写成 invisible，但保留 collision / proxy。

推荐打开：

```text
*_nurec_with_collision.usdz
```

---

### 11.5 只想调试 SOG 解码，不想转 USDZ

使用：

```bash
--skip-usdz
```

示例：

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root /path/to/lcc2-result \
  --output-dir /tmp/lcc2-debug \
  --lod-depth highest \
  --skip-usdz \
  --force
```

这会输出：

```text
*_gaussians.ply
*_mesh.ply
*_summary.json
```

---

### 11.6 输出文件名不知道是什么

查看 summary：

```bash
cat /workspace/data/lcc2_from_sog_tool/*_summary.json
```

重点看：

```json
"gaussians_ply": "...",
"mesh_ply": "...",
"nurec_usdz": "...",
"collision_usdz": "..."
```

---

## 12. 当前项目已验证命令

当前项目中已验证的完整转换命令如下：

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root "/workspace/data/lcc2_sog_src/3D模型/lcc2-result" \
  --output-dir /workspace/data/lcc2_from_sog_tool \
  --lod-depth highest \
  --threedgrut-python /tmp/3dgrut_src/.venv/bin/python \
  --collision \
  --force
```

已验证输出：

```text
/workspace/data/lcc2_from_sog_tool/444ab7c3911871d73a84e531866561ff_lod4_nurec_with_collision.usdz
```

对应 summary：

```text
/workspace/data/lcc2_from_sog_tool/444ab7c3911871d73a84e531866561ff_summary.json
```

Isaac Lab 打开命令：

```bash
bash scripts/_internal/entry/open_substation_usdz_gui.sh \
  /workspace/data/lcc2_from_sog_tool/444ab7c3911871d73a84e531866561ff_lod4_nurec_with_collision.usdz
```

---

## 13. 推荐调试流程

建议不要一上来就完整转 USDZ。

推荐流程如下：

### 第一步：确认基础工具可运行

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py --help
```

### 第二步：只导出 PLY

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root "/workspace/data/lcc2_sog_src/3D模型/lcc2-result" \
  --output-dir /workspace/data/lcc2_from_sog_tool \
  --lod-depth highest \
  --skip-usdz \
  --force
```

检查：

```bash
cat /workspace/data/lcc2_from_sog_tool/*_summary.json
```

确认：

```text
selected_gaussians 正常
gaussian_bbox_min/max 正常
```

### 第三步：测试 3DGRUT 环境

```bash
/tmp/3dgrut_src/.venv/bin/python -c "import threedgrut; print('3DGRUT OK')"
```

### 第四步：完整导出 USDZ

```bash
python tools/lcc2-to-nurec-usdz/lcc2_to_nurec_usdz.py \
  --input-root "/workspace/data/lcc2_sog_src/3D模型/lcc2-result" \
  --output-dir /workspace/data/lcc2_from_sog_tool \
  --lod-depth highest \
  --threedgrut-python /tmp/3dgrut_src/.venv/bin/python \
  --collision \
  --force
```

### 第五步：在 Isaac Lab 中打开

```bash
bash scripts/_internal/entry/open_substation_usdz_gui.sh \
  /workspace/data/lcc2_from_sog_tool/444ab7c3911871d73a84e531866561ff_lod4_nurec_with_collision.usdz
```

---

## 14. 参考资料

* XGRIDS LCC2 Whitepaper: https://github.com/xgrids/LCC2Whitepaper
* XGRIDS SOG reader implementation: https://github.com/xgrids/splat-transform
* NVIDIA 3DGRUT: https://github.com/nv-tlabs/3dgrut
