# ⚡ fused-turboquant

[![PyPI](https://img.shields.io/pypi/v/fused-turboquant?color=blue)](https://pypi.org/project/fused-turboquant/)
[![Python](https://img.shields.io/pypi/pyversions/fused-turboquant)](https://pypi.org/project/fused-turboquant/)
[![License](https://img.shields.io/github/license/Argonaut790/fused-turboquant)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b)](https://arxiv.org/abs/2504.19874)
[![GitHub stars](https://img.shields.io/github/stars/Argonaut790/fused-turboquant?style=social)](https://github.com/Argonaut790/fused-turboquant)

**fused-turboquant** is a high-performance Python library for compressing LLM KV caches to 2-4 bits using the TurboQuant algorithm (Google Research, ICLR 2026). It fuses the entire quantization pipeline — Randomized Hadamard Transform, normalization, Lloyd-Max quantization, and bit packing — into single Triton GPU kernels, achieving up to 4.9x memory compression with near-lossless quality. Drop-in support for HuggingFace Transformers and vLLM. Works with Llama, Qwen, Mistral, Phi, and more.

- 🗜️ Compresses both **K and V** caches to **2-4 bits** using [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) — up to **~4.9x KV cache compression** (3-bit) or **~3.8x** (4-bit)
- 🔥 Fuses the entire encode/decode pipeline into **single Triton kernels** (1 kernel vs 5+ in other implementations)
- 🦋 Uses **RHT** instead of dense QR rotation — O(d log d) compute, O(d) storage, fits in registers
- 🤗 Drop-in **HuggingFace** integration and **vLLM** attention backend
- 🔄 Auto-detects CUDA + Triton; falls back to unfused PyTorch on CPU

## 📦 依存関係インストール

```bash
# HuggingFace 経由で使う場合
pip install "fused-turboquant[cuda,hf]"

# vLLM 経由で使う場合
pip install "fused-turboquant[cuda,vllm]"

# 両方
pip install "fused-turboquant[cuda,hf,vllm]"
```

> ⚠️ `torch.cuda.is_available()` が `False` を返す環境では先に CUDA 版 PyTorch を入れてください: `pip install torch --index-url https://download.pytorch.org/whl/cu128`

ソースからインストールする場合:

```bash
git clone https://github.com/Argonaut790/fused-turboquant.git
cd fused-turboquant
pip install -e ".[dev]"   # 開発依存も含めて editable install
```

## 🚀 Quick Start

### 🤗 HuggingFace Transformers

`patch_model` がモデルの全 attention 層に圧縮 KV キャッシュを差し込みます。RoPE / GQA は自動検出されます。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fused_turboquant.hf import patch_model

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# K+V を 4-bit で圧縮 (RHT 回転 + Lloyd-Max 量子化 + bit pack を 1 つの Triton カーネルに融合)
cache = patch_model(model, bits=4)

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=50,
    use_cache=True,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

オプション (`patch_model` 引数):
- `bits=3`: 圧縮率を 4.9 倍まで上げる (精度はやや低下)
- `compress_v=False`: K のみ圧縮 (V は FP16)
- `compress_v="boundary"`: 最初と最後の 2 層を FP16 のまま (精度優先)
- `quantizer_kind="planar"` / `"rotor"` / `"iso_fast"` / `"iso_full"`: ブロック対角型回転 (デフォルト `"rht"`)

### 🔌 vLLM

Python API (オフラインバッチ推論)。 ハイパラは Gemma 4 31B Instruct での 300 サンプル GSM-8K 評価で精度劣化 −0.34 pts (97.67% → 97.33%)、 KV cache 9.7× 圧縮を達成した構成 (rht + BP=0 + V_ROTATE + K3V3) を採用:

```python
import os
os.environ["TURBOQUANT_KIND"] = "rht"          # 回転種別 (rht/planar/rotor/iso_fast/iso_full)
os.environ["TURBOQUANT_BOUNDARY_PROTECT"] = "0" # rht は FWHT で boundary 層も gaussian 化されるので BP 不要
os.environ["TURBOQUANT_V_ROTATE"] = "1"        # V も K と同じ rotation で量子化 (rotorquant 流) — 精度確保に必須

import fused_turboquant.vllm_plugin  # noqa: F401  (TURBOQUANT バックエンドを登録)
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    dtype="bfloat16",
    attention_backend="TURBOQUANT",
    kv_cache_dtype="turboquant_3bit_nc",   # 3-bit K + 3-bit V (4-bit 比 KV cache -23%)
    enforce_eager=False,                   # CUDA graphs ON (decode を 5–14 倍高速化)
)
sp = SamplingParams(max_tokens=128, temperature=0.0)
out = llm.generate(["The capital of France is"], sp)
print(out[0].outputs[0].text)
```

OpenAI 互換 API サーバとして起動する場合:

```bash
TURBOQUANT_KIND=rht TURBOQUANT_BOUNDARY_PROTECT=0 TURBOQUANT_V_ROTATE=1 \
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --attention-backend TURBOQUANT \
  --kv-cache-dtype turboquant_3bit_nc
```

備考:
- **`BP=0` は `TURBOQUANT_KIND=rht` 限定で安全**。 block-diag 系 (planar/rotor/iso_fast/iso_full) は boundary 層を gaussian 化しきれないので `BP=fp8` か `BP=1` を併用してください (詳細は次節)。
- **`TURBOQUANT_V_ROTATE=1` は全 quant 構成で精度に効きます** (Gemma 4 31B + rotor で +4.3〜+72 pts の改善)。 計算オーバーヘッドは per-decode-step の `output @ M^T` GEMM 1 回 (≪ 0.1 ms)。
- さらに圧縮したい場合は K3V2 / K2V2 (`TURBOQUANT_KEY_BITS=2 TURBOQUANT_VALUE_BITS=2`) で −2.67 pts / 14.6× 圧縮まで実用領域。 各モードの効果と推奨設定は [次節 (🎛️ vLLM 性能モード一覧)](#%EF%B8%8F-vllm-性能モード一覧) を参照。

## 🎛️ vLLM 性能モード一覧

vLLM バックエンドには 3 つの独立した切り替えスイッチがあり、これらを組み合わせることで decode 速度・圧縮率・対応 prefix 長を調整できます。ワークロードに合った行を選んでください。

| モード | 環境変数 / フラグ | 効果 | 推奨用途 |
|---|---|---|---|
| **CUDA graphs** | `enforce_eager=False` *(vLLM)* | decode 1 ステップごとの kernel 起動 overhead を償却 — 小型モデルで **decode 5〜14 倍**、30B+ で約 1.7 倍 | 本番では **常に ON** |
| **Boundary protection (境界層保護)** | `TURBOQUANT_BOUNDARY_PROTECT=1\|fp16\|2\|3\|4\|8` | 最初と最後の 2 層を **指定 bit 数で** 保持。`1` / `fp16` (デフォルト) = FP16 raw; `2`〜`4` / `8` = TQ 量子化の bit 数を上書き (slot は FP16 サイズ確保のままで余裕あり)。精度向上と同時に、上位 bit (FP16 / 8-bit) の層が centroid lookup より速く attention できるため throughput も BP=0 を上回ることが多い。`defer_prefill` を CUDA graph で正しく動かすために **必須** | 圧縮率を限界まで詰めたい場合を除き **常に ON** |
| **Deferred FP16 K-cache (遅延 K キャッシュ量子化)** | `TURBOQUANT_DEFER_PREFILL=1` | prefill トークンの回転・量子化をスキップし、サイドバッファに FP16 のまま保持。decode 時に 2 段階 attention (FP16 prefix 上で flash-attn + 量子化領域上で offset-decode kernel、log-sum-exp で merge) を実行 | **長コンテキスト (prefix ≥ 4〜5K tokens) のみ**。それ未満では遅くなる (速度表参照) |
| **回転種別** | `TURBOQUANT_KIND=rht\|planar\|rotor\|iso_fast\|iso_full` | MSE 量子化の前に適用する直交変換を選択。RHT (Walsh-Hadamard butterfly) が本番標準、Planar / Rotor / Iso はブロック対角型 (2×2 / 3×3 / 4×4) で各々 in-kernel 回転実装あり | 精度重視は RHT、rotorquant 論文派生は Planar / Iso |

### ワークロード別 推奨設定

| ワークロード | `enforce_eager` | `TURBOQUANT_BOUNDARY_PROTECT` | `TURBOQUANT_DEFER_PREFILL` |
|---|---|---|---|
| 短い chat (prefix ≤ 4K tokens) — **既定** | `False` | `1` | `0` |
| 長コンテキスト RAG / agent (prefix ≥ 5K tokens) | `False` | `1` | `1` |
| 圧縮率優先 / decode が遅くてもよい場合 | `False` | `0` | `0` |

### 実測 decode 速度 (Qwen 2.5-3B-Instruct, RTX PRO 6000, planar)

各設定を切り替えたときの decode スループットを示します。プロンプト長は **入力トークン数 (tokens)**、表中の数値は **decode ステップあたりのトークン生成速度 (tok/s)** です。**太字** は各行の最速設定。

| プロンプト長 [tokens] | Eager + BP=0 [tok/s] | + CUDA graphs [tok/s] | + BP=1 [tok/s] | + defer (BP=1) [tok/s] |
|---:|---:|---:|---:|---:|
| 181 | 35.5 | 194.7 | **193.2** | 157.7 |
| 1,201 | 22.5 | 181.8 | **181.8** | 157.1 |
| 4,801 | 10.4 | 147.4 | 150.2 | **155.3** |
| 19,201 | 38.6 | 88.7 | 93.8 | **155.6** |

### カーネル単体の速度 (RHT vs 各ブロック対角型, RTX PRO 6000)

fused store カーネル単体 (rotation + bucketize + V quant + slot scatter まで含む 1 launch) を計測したものです。`matrix(D,D)` は最初に実装した汎用 `(D,D)` 行列を SRAM タイルでまるごと matmul する経路で、ブロック対角型の理論優位性を計測する比較基準として残しています (実運用では block-diag 戦略がデフォルト)。1024 トークン × KV 8 ヘッド × バッチ 1。

| 設定 | RHT (FWHT butterfly) [µs] | matrix(D,D) [µs] | planar (B=2) [µs] | rotor (B=3) [µs] | iso_fast (B=4) [µs] | iso_full (B=4) [µs] |
|---|---:|---:|---:|---:|---:|---:|
| **prefill 1024 tok, D=128** | 100.3 | 114.4 | **100.2** | **99.9** | **99.5** | 100.3 |
| **prefill 1024 tok, D=256** | 106.6 | **3,088.3** | **100.0** | **100.4** | **100.2** | **99.8** |
| **decode 1 tok, D=128** | 100.7 | 88.9 | 100.4 | 100.1 | 100.4 | 100.7 |
| **decode 1 tok, D=256** | 99.6 | 92.6 | 101.8 | 101.0 | 100.8 | 100.1 |

ポイント:
- **D=256 prefill**: 汎用 `(D,D)` matmul カーネルは 65 KB の float32 タイルが SRAM に乗らずタイル化が走るため **約 30 倍遅く** なる。これに対し block-diag (Planar / Rotor / Iso) はブロックあたり 4〜16 FMA で済むため約 100 µs に揃う。これが [rotorquant の論文](https://www.scrya.com/rotorquant/)が主張する「TurboQuant 比 10〜19 倍速い」の正体で、本実装でも同等の効果が得られています。
- **D=128 prefill**: `(D,D)` matmul も SRAM に乗るので差は小さい (約 14%)。RHT / block-diag / matrix のいずれもほぼ同じ。
- **decode 1 token**: 8 ヘッド × 1 トークンとワークが小さく、すべてのカーネルが約 100 µs で頭打ち = **CUDA launch overhead 律速**。実プログラムを 1 launch 単位で見ると差はほぼ noise レベルなので、end-to-end の decode スループットには直接効きません。
- block-diag 戦略間 (Planar / Rotor / Iso_Fast / Iso_Full) は B=2〜4 で FMA 数こそ違うものの、measure 上は全て約 100 µs に揃う (B が小さいほど僅かに速いがやはり launch 律速)。**精度の差はあっても速度の差は微小**、というのが実装上の結論です。

### K と V に独立な bit 数を割り当てる

vLLM 側の `kv_cache_dtype` プリセット (Pydantic Literal で固定) を変えずに、**`K, V ∈ {2, 3, 4}` の全 9 組み合わせ**を選べるようにしてあります。仕組み:

- `kv_cache_dtype` プリセットは **スロット領域の確保サイズ**を決める。
- 環境変数 **`TURBOQUANT_KEY_BITS` / `TURBOQUANT_VALUE_BITS`** で **実効的な量子化 bit 数**を独立に上書き。
- 制約: override の bit 数 ≤ プリセットの bit 数 (スロットがあふれないように)。全 9 組み合わせを使いたい時は `turboquant_4bit_nc` を選び、そこから K / V を 2〜4 で自由に指定。

| 目的 (K, V) | `kv_cache_dtype` の最小選択肢 | 環境変数の例 |
|---|---|---|
| K=4, V=4 | `turboquant_4bit_nc` | (override 不要) |
| K=4, V=3 | `turboquant_4bit_nc` | `TURBOQUANT_VALUE_BITS=3` |
| K=4, V=2 | `turboquant_4bit_nc` | `TURBOQUANT_VALUE_BITS=2` |
| K=3, V=4 | `turboquant_k3v4_nc` または `turboquant_4bit_nc` | (前者は override 不要 / 後者は `TURBOQUANT_KEY_BITS=3`) |
| K=3, V=3 | `turboquant_3bit_nc` または `turboquant_4bit_nc` | (前者は override 不要) |
| K=3, V=2 | `turboquant_k3v4_nc` または `turboquant_4bit_nc` | `TURBOQUANT_KEY_BITS=3 TURBOQUANT_VALUE_BITS=2` |
| K=2, V=4 | `turboquant_4bit_nc` | `TURBOQUANT_KEY_BITS=2` |
| K=2, V=3 | `turboquant_4bit_nc` | `TURBOQUANT_KEY_BITS=2 TURBOQUANT_VALUE_BITS=3` |
| K=2, V=2 | `turboquant_3bit_nc` または `turboquant_4bit_nc` | `TURBOQUANT_KEY_BITS=2 TURBOQUANT_VALUE_BITS=2` |

#### 例: K=2 / V=4 (K だけ最大圧縮、V は精度温存)

```bash
TURBOQUANT_KEY_BITS=2 \
TURBOQUANT_BOUNDARY_PROTECT=1 \
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --attention-backend TURBOQUANT \
  --kv-cache-dtype turboquant_4bit_nc
```

または Python から:

```python
import os
os.environ["TURBOQUANT_KEY_BITS"] = "2"
os.environ["TURBOQUANT_VALUE_BITS"] = "4"
os.environ["TURBOQUANT_BOUNDARY_PROTECT"] = "1"
import fused_turboquant.vllm_plugin  # noqa: F401
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", dtype="bfloat16",
          attention_backend="TURBOQUANT",
          kv_cache_dtype="turboquant_4bit_nc")
```

実装上の補足:

- 2-bit K / V は本リポジトリの `_bucketize_pack_norm_v_store` (store kernel) と `_tq_decode_stage1_offset` (decode kernel fork) に 2-bit 専用 branch を追加して対応。**vLLM の stock decode kernel (`_tq_decode_stage1`) は 2-bit 非対応**のため、`TURBOQUANT_KEY_BITS=2` または `TURBOQUANT_VALUE_BITS=2` の場合は自動的に本リポジトリの fork を経由します。
- Lloyd-Max centroids は vLLM の `get_centroids(d, bits=2)` をそのまま使用 (4 levels)。
- スロットの ceiling サイズはプリセット由来なので、override で bit 数を下げてもメモリ削減は **プリセット由来の slot 内** に閉じます (例: K=2 を 4-bit slot 上で動かすと 2-bit 分のデータ + 2-bit 分の未使用領域)。本当の memory saving が必要な場合は `kv_cache_dtype` を `turboquant_3bit_nc` 等小さい preset に。
- FP8 keys (`turboquant_k8v4`) は v1 backend では未サポート (`turboquant_4bit_nc` / `k3v4_nc` / `3bit_nc` の 3 種から選択)。

### 境界層保護の bit 数を指定する (FP16 以外の量子化で境界層を保護)

境界層保護 (Boundary Protection) は従来「最初と最後の 2 層を FP16 で保持」する機能でしたが、`TURBOQUANT_BOUNDARY_PROTECT` を bit 数で指定すると **境界層を任意の bit 数で量子化** できます。FP16 (16-bit) より大幅にメモリ削減しつつ、中間層 (3-bit / 4-bit) より高精度に保つ、という中間の選択肢を得られます。

| `TURBOQUANT_BOUNDARY_PROTECT` | 境界層の扱い | スロット使用バイト数 [bytes] (head_dim=128 時) |
|---|---|---:|
| `0` / `off` | 保護なし (全層中間層と同じ bit 数) | (中間層と同じ) |
| `1` / `fp16` (既定) | 境界層を **FP16 raw** で保持。flash-attn / SDPA 経由 | 512 (4·D = K_fp16 + V_fp16) |
| `2` | 境界層を **2-bit** TQ で保持 | 70 (32 K + 2 norm + 32 V + 4 V scale/zero) |
| `3` | 境界層を **3-bit** TQ で保持 | 102 (48 K + 2 + 48 V + 4) |
| `4` | 境界層を **4-bit** TQ で保持 | 134 (64 K + 2 + 64 V + 4) |
| `8` | 境界層を **8-bit** TQ (256 levels MSE K + 256 levels uniform V) で保持 | 262 (128 K + 2 + 128 V + 4) |

スロット自体は FP16-sized (= `4·head_dim` bytes) のまま確保されており、上記いずれの bit 数でも余裕で fit します (`head_dim=128` で 512 bytes 確保、最大の 8-bit 設定でも 262 bytes 使用)。残りは未使用領域。

#### 使用例

```bash
# 境界層を 8-bit TQ で保護 (中間層は 4-bit のまま)
TURBOQUANT_BOUNDARY_PROTECT=8 \
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --attention-backend TURBOQUANT \
  --kv-cache-dtype turboquant_4bit_nc
```

#### 実装メモ

- 境界層は vLLM の `kv_cache_dtype_skip_layers` 機構によって `cache_dtype="auto"` (FP16 spec) で確保されます。境界層保護を bit 数指定にした場合も、**スロット自体は FP16-sized のまま**で、TQ kernels が先頭の数十〜数百バイトのみを使用します。残りは inert (読み書きされません)。
- 8-bit boundary は 256 Lloyd-Max centroids + 256 levels uniform V。本リポジトリの fork 版 decode kernel に `VQB == 8` branch を追加して実装。`MSE_BITS == 8` 側は既存 kernel の汎用 bit-shift formula で自動的にカバーされます。
- 2-bit / 3-bit / 4-bit boundary は既存の middle-layer 用の同 bit-width 経路を流用します。
- `TURBOQUANT_KEY_BITS` / `TURBOQUANT_VALUE_BITS` (中間層用) は境界層には作用しません。境界層は `TURBOQUANT_BOUNDARY_PROTECT` の bit 数で決まります (K, V 同 bit のみ; K/V 個別指定は今後の課題)。

補足:
- 比較対象: [rotorquant](https://github.com/scrya-com/rotorquant) は llama.cpp 上の Qwen 2.5-3B planar3 で 119 tok/s と報告。本実装では同条件で 193 tok/s、19K トークン prefix でも 155 tok/s を維持します。
- `TURBOQUANT_DEFER_PREFILL=1` を使うには `TURBOQUANT_BOUNDARY_PROTECT=1` が必要です。BP=0 + defer + cudagraph はクラッシュせず動きますが、capture 時のディスパッチが不一致のため出力が無音で壊れます。
- defer 経路は層ごとに `[max_model_len, num_kv_heads, head_size]` の FP16 サイドバッファを確保するため、量子化 paged cache に加えてメモリを消費します。
- `head_size > 256` のモデル (例: Gemma 4 31B のグローバル層) では境界層が flash-attn ではなく CUDA graph 安全な SDPA gather に fallback します。CUDA graphs 自体は動きますが、eager 比の高速化倍率は ~1.7 倍 (5×+ ではなく) になります。

## 📦 Installation

```bash
pip install fused-turboquant[cuda]          # core + Triton fused kernels
pip install fused-turboquant[cuda,hf]       # + HuggingFace transformers
pip install fused-turboquant[vllm]          # + vLLM plugin
pip install fused-turboquant                # core only (torch + scipy + numpy)
```

> ⚠️ If `torch.cuda.is_available()` returns `False`, install CUDA-enabled PyTorch first:
> `pip install torch --index-url https://download.pytorch.org/whl/cu128`

**From source** (development):

```bash
git clone https://github.com/Argonaut790/fused-turboquant.git
cd fused-turboquant
pip install -e ".[dev]"       # or: uv sync --extra dev
```

## 🚀 Quick Start

```python
import torch
from fused_turboquant import TurboQuantMSE

tq = TurboQuantMSE(head_dim=256, bits=4, device="cuda")
keys = torch.randn(1, 4, 128, 256, device="cuda")

compressed = tq.encode(keys)   # 1 fused Triton kernel
decoded = tq.decode(compressed) # 1 fused Triton kernel

print(f"Compression: {compressed.compression_ratio:.1f}x")  # 3.9x
```

## 🛠️ Usage

### 🤗 HuggingFace Integration

Requires `pip install fused-turboquant[cuda,hf]`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fused_turboquant.hf import patch_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

cache = patch_model(model, bits=4)  # compresses K+V cache in all attention layers
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=50, use_cache=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

`patch_model` compresses both keys and values via fused Triton encode. Keys are stored packed (nibble-packed for 4-bit) and Q·K^T is computed directly from packed indices with inline unpacking — no dequantization pass. Values are decompressed on the fly during the attention-weighted sum. Supports `compress_v=False` (K-only), `compress_v="boundary"` (first/last 2 layers at fp16), or a custom `callable(layer_idx, n_layers) -> bool`.

### 🔌 vLLM Integration

Serve any supported model with compressed KV cache through vLLM's standard serving stack — continuous batching, OpenAI-compatible API, and paged block management all work transparently with compressed blocks.

```bash
pip install fused-turboquant[vllm]
```

**Online serving** (OpenAI-compatible API):

```bash
vllm serve Qwen/Qwen3-8B --attention-backend FUSED_TURBOQUANT
```

**Offline batch inference** (Python API):

```python
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-8B", attention_backend="FUSED_TURBOQUANT")
outputs = llm.generate(
    ["Explain KV cache compression in one paragraph."],
    SamplingParams(max_tokens=200, temperature=0.0),
)
print(outputs[0].outputs[0].text)
```

**Configuration** via environment variables:

| Variable | Default | Description |
|---|---|---|
| `TURBOQUANT_BITS` | `4` | Quantization bit-width (2, 3, or 4) |
| `TURBOQUANT_COMPRESS_V` | `1` | Compress values too (`1`) or K-only (`0`) |

```bash
# 3-bit K+V compression for maximum memory savings
TURBOQUANT_BITS=3 vllm serve Qwen/Qwen3-8B --attention-backend FUSED_TURBOQUANT
```

The plugin auto-registers via Python entry points — no code changes needed to vLLM. KV cache blocks are stored as packed uint8 indices + fp32 norms within vLLM's paged block system. Prefill uses FlashAttention/SDPA on full-precision KV, then compresses for storage. Decode uses our fused Triton QK kernel directly on packed indices.

**Compatibility**: vLLM 0.8 – 0.18+ | Requires CUDA with Triton

## ✅ Supported Models

### Requirements

For compression to apply, a model's attention layers must satisfy:

- 🔢 **`head_dim` is a power of 2** (64, 128, 256) — required by the Randomized Hadamard Transform
- 🎛️ **`bits` is 2, 3, or 4** — Lloyd-Max codebooks are precomputed for these bit-widths
- 🖥️ **CUDA + Triton** for fused kernels (unfused PyTorch fallback on CPU for encode/decode, but not fused attention)

`patch_model` (fused cache) additionally requires:

- 🔑 **Separate QKV projections** — `q_proj`, `k_proj`, `v_proj` (not fused `qkv_proj` or `c_attn`)
- 🔄 **RoPE via `position_embeddings=(cos, sin)`** — the convention used by Llama, Qwen, and most modern models
- ➗ **`n_q_heads` divisible by `n_kv_heads`** — standard for GQA/MQA models
- 🚫 **No sliding window or logit softcapping** — these are validated and rejected upfront

### Model Compatibility

> 📋 `head_dim` values verified from official HuggingFace `config.json` files (March 2026).

#### Dense Decoder-Only

| Model | `head_dim` | `patch_model` | Notes |
|:------|:----------:|:-------------:|:------|
| 🦙 **Llama 3.1/3.2/3.3** (1B–405B) | 128 | ✅ Tested | Standard GQA + RoPE |
| 🟣 **Qwen3** (0.6B–235B) | 128 | ✅ Tested | GQA + RoPE |
| 🟣 **Qwen2.5** (0.5B–72B) | 128 | ✅ Tested | |
| 🔷 **Mistral 7B / Small 3** | 128 | ✅ Expected | Standard GQA + RoPE |
| 🔵 **Phi-4** (14B) | 128 | ✅ Expected | `hidden=5120 / 40 heads = 128` |
| 🟢 **Command R / R+** | 128 | ✅ Expected | |
| 🟢 **Yi** (1.5, 6B–34B) | 128 | ✅ Expected | |
| 🟢 **InternLM 2 / 3** | 128 | ✅ Expected | |

#### MoE (Mixture of Experts)

MoE models share attention with their dense variants — only FFN layers use expert routing. fused-turboquant patches attention only, so MoE routing is completely unaffected.

| Model | `head_dim` | `patch_model` | Notes |
|:------|:----------:|:-------------:|:------|
| 🟣 **Qwen3-MoE** (30B-A3B, 235B-A22B) | 128 | ✅ Expected | Same attention as Qwen3 dense |
| 🔷 **Mixtral** (8x7B, 8x22B) | 128 | ✅ Expected | Same attention as Mistral 7B |
| 🟢 **OLMoE** (6.9B, 1.3B active) | 128 | ✅ Expected | Allen AI; standard GQA |
| 🟢 **Zen4-Max** (30B, 3B active) | 128 | ✅ Expected | Apache 2.0 |

#### Multimodal (Vision-Language)

For multimodal models, fused-turboquant **patches only the text decoder** attention layers. Vision encoder attention is automatically skipped by module-path detection (`visual`, `vision_model`, `vision_tower`, etc.).

| Model | `head_dim` | `patch_model` | Notes |
|:------|:----------:|:-------------:|:------|
| 🟣 **Qwen2-VL** | 128 | ✅ Expected | Vision encoder skipped; text decoder patched |
| 🟢 **InternVL 2/3** | 128 | ✅ Expected | Vision encoder skipped; text decoder patched |
| 🟢 **LLaVA** (Llama/Vicuna backbone) | 128 | ✅ Expected | CLIP ViT skipped (different naming) |

#### Hybrid Architectures

Models mixing full attention with linear attention (DeltaNet, Mamba, etc.). Only the full-attention layers are patched; linear layers are auto-skipped.

| Model | `head_dim` | `patch_model` | Notes |
|:------|:----------:|:-------------:|:------|
| 🟣 **Qwen3.5-MoE** (35B-A3B, 397B-A17B) | 256 | ✅ Expected | 10 of 40 layers patched (every 4th = full attn); DeltaNet + MoE layers skipped |

#### Not Compatible

| Model | Reason |
|:------|:-------|
| 🌊 **DeepSeek V3 / V3.2 / V4** | Multi-Latent Attention (MLA) — no standard QKV projections |
| 🦙 **Llama 4** Scout / Maverick | QK norm (`use_qk_norm=true`) + chunked attention — both unsupported |
| 💎 **Gemma 3** (1B–27B) | Alternating sliding window / full attention — sliding window unsupported |
| 🔷 **Mistral Small 4** (119B MoE) | Sliding window attention — unsupported |
| 🔵 **Phi-3 / 3.5** (mini, small) | Fused `qkv_proj` + `head_dim=96` (not power of 2) |
| 🟢 **GPT-2 / BLOOM / Falcon** | No RoPE; fused QKV projection (`c_attn` / `query_key_value`) |
| 🔴 **NVIDIA Nemotron 3 Super** | Hybrid Mamba-Transformer MoE — Mamba layers not patchable |

**Legend**: ✅ Tested = verified end-to-end, ✅ Expected = compatible architecture (not yet tested), ❌ = incompatible.

> 🔧 **Smart detection**: `patch_model` validates each layer upfront and raises clear errors for unsupported features (sliding window, fused QKV, logit softcapping, cross-attention). QK layer norm (Qwen3, Gemma3) is fully supported. It also runs a smoke test to catch any silent issues. Use `verify=False` to skip the smoke test.
>
> 🔍 **Vision-safe**: For multimodal models, attention layers inside vision encoders (`visual.*`, `vision_model.*`, `vision_tower.*`) are automatically skipped — only the text decoder is patched.

**🔍 Check before patching** — use `check_model_compatibility()` to diagnose any model:

```python
from fused_turboquant.hf import check_model_compatibility

result = check_model_compatibility(model)
print(result)
# {'compatible': True, 'head_dim': 128, 'eligible_layers': 24,
#  'rope_detected': True, 'unsupported_features': [],
#  'vision_layers_skipped': 0, 'known_compatible': True, ...}
```

## 🧠 How It Works

TurboQuant compresses KV vectors by rotating them into a uniform distribution, then quantizing each coordinate independently. Both keys and values go through the same pipeline:

```
Encode: input → RHT rotate → normalize → Lloyd-Max quantize → pack nibbles
Decode: unpack → dequantize → denormalize → inverse RHT → output
```

**Why fusion matters**: Other implementations use dense QR rotation (O(d²) matrix, 256 KB at d=256), which forces a separate cuBLAS matmul and prevents kernel fusion. RHT needs only a d-element sign vector (1 KB) that fits in SRAM, so the entire pipeline runs in **one kernel launch** with zero HBM round-trips between stages:

```
Other implementations:  [rotation] → HBM → [norm] → HBM → [quantize] → HBM → [pack]    5+ kernels
This project:           [single Triton kernel: RHT → norm → quantize → pack]              1 kernel
```

**Prefill vs Decode**: Compression applies to the **decode (generation) phase**. During prefill (processing the prompt), we use Flash Attention (SDPA) on full FP16 keys and values for maximum speed — the KV tensors are then compressed and stored in packed form. During autoregressive decode, each new token's KV is compressed on arrival, and the fused Triton kernel computes Q·K^T directly from packed indices. The memory savings come from the **stored KV cache** being compressed — this is what dominates memory at long contexts.

## 📊 Benchmarks

All benchmarks: NVIDIA GB10 (Blackwell, unified memory), Qwen3-8B, PyTorch 2.10+cu128.

### Memory Savings

**The primary benefit of TurboQuant KV cache compression is memory reduction** — fitting longer contexts and larger batches in the same GPU memory.

#### Compression Ratio

Both K and V caches are compressed by default (`compress_v=True`). All bit-widths use true packed storage — nibble-packed for 4-bit, bitstream-packed for 3-bit (8 values per 3 bytes), 2-bit packed for 2-bit. Keys are unpacked inline by the fused attention kernel (shift+mask, no separate dequant pass).

| Config | Nominal bits | Effective bits/elem | KV Compression | Per-position storage (head_dim=128) |
|--------|:---:|:---:|:---:|:---|
| FP16 baseline | 16 | 16.0 | 1.0x | K: 256B + V: 256B = 512B |
| **FusedTQ4** (K+V) | 4 | 4.25 | **~3.8x** | K: 68B + V: 68B = 136B |
| **FusedTQ3** (K+V) | 3 | 3.25 | **~4.9x** | K: 52B + V: 52B = 104B |

> **KV cache compression vs total memory**: The ~3.8x / ~4.9x ratios above are for the **KV cache data only** (512B → 136B or 104B per position). Total GPU memory also includes model weights (~16 GB for Qwen3-8B) which are unchanged by KV compression. At short contexts the KV cache is small relative to model weights, so end-to-end savings appear modest; at long contexts the KV cache dominates and savings approach the theoretical ratios:
>
> | Context | KV Cache Share of Total | End-to-End Memory Saved (FusedTQ4) |
> |--------:|:-----------------------:|:----------------------------------:|
> | 4K | ~3% | ~418 MB (2.5%) |
> | 32K | ~30% | ~3,348 MB (14%) |
> | 128K+ | >50% | Approaches ~3.8x |
>
> This is why KV cache compression becomes critical at scale — at 1M tokens the KV cache alone would be ~128 GB in FP16 but only ~34 GB with FusedTQ4.
>
> **Effective bit-rate**: The nominal `--bits` is the index width; the per-vector fp32 norm adds overhead. For head_dim=128: `bits=3` → 3.25 bits/elem, `bits=4` → 4.25 bits/elem.
>
> **Layer-aware V compression**: Use `compress_v="boundary"` to keep the first 2 and last 2 layers at fp16 V precision while compressing the rest. This can recover quality on sensitive models with negligible memory cost. Custom per-layer strategies are also supported via `compress_v=callable`.

#### Measured End-to-End Peak Memory

Peak GPU memory (model weights + KV cache + buffers) during generation on NVIDIA GB10. Compression applies to the **decode phase** — prefill uses Flash Attention on full FP16, then KV is compressed for storage. Note: model weights (~16 GB) are constant overhead — the savings below come entirely from the compressed KV cache portion.

Qwen3-8B, K+V compressed, 50 decode tokens per context length:

**3-bit (FusedTQ3):**

| Context | FP16 Peak | FusedTQ3 K+V (ours) | Saved | TQ3 (Dejan.ai) |
|--------:|:---------:|:-------------------:|:-----:|:---------------:|
| 4,096 | 16,626 MB | **16,179 MB** | 447 MB | 17,630 MB |
| 8,192 | 17,620 MB | **16,726 MB** | 894 MB | 18,625 MB |
| 16,384 | 19,608 MB | **17,790 MB** | 1,818 MB | 20,613 MB |
| 32,768 | 23,585 MB | **19,949 MB** | 3,636 MB | 24,590 MB |

**4-bit (FusedTQ4):**

| Context | FP16 Peak | FusedTQ4 K+V (ours) | Saved | TQ4 (Dejan.ai) |
|--------:|:---------:|:-------------------:|:-----:|:---------------:|
| 4,096 | 16,626 MB | **16,208 MB** | 418 MB | 17,913 MB |
| 8,192 | 17,620 MB | **16,783 MB** | 837 MB | 18,907 MB |
| 16,384 | 19,608 MB | **17,934 MB** | 1,674 MB | 20,895 MB |
| 32,768 | 23,585 MB | **20,237 MB** | 3,348 MB | 24,872 MB |

> **Note**: Dejan TQ uses *more* memory than FP16 because it materializes decompressed keys for standard attention. FusedTQ avoids this entirely by computing Q·K^T directly from packed compressed indices.

#### Projected KV Cache at 1M Context

Based on the measured per-token compression ratios, the KV cache (excluding model weights) at extreme context lengths for Qwen3-8B (32 layers, 8 KV heads, head_dim=128):

| Context | FP16 KV Cache | FusedTQ4 K+V (~3.8x) | FusedTQ3 K+V (~4.9x) |
|--------:|:-------------:|:---------------------:|:---------------------:|
| 131,072 (128K) | 16 GB | 4.2 GB | **3.2 GB** |
| 524,288 (512K) | 64 GB | 17 GB | **13 GB** |
| **1,048,576 (1M)** | **128 GB** | **34 GB** | **26 GB** |

> *Calculated from measured per-token storage: FP16=128 KB/token, FusedTQ4=34 KB/token, FusedTQ3=26 KB/token.* At 1M context, FP16 KV cache alone exceeds most GPU VRAM (128 GB). FusedTQ3 reduces this to **26 GB** — fitting on a single 80 GB GPU (A100/H100).

### Quality

Measured on Qwen3-8B via teacher-forced decode: 20 tokens prefilled, then ground-truth tokens fed one at a time through the compressed KV attention path. Logit cosine similarity compared against FP16 reference at each decode position.

| Config | Avg Logit Cosine Sim | Min Logit Cosine Sim | Quality Impact |
|--------|:---:|:---:|:---|
| FusedTQ4, K+V | **0.9869** | 0.9048 | Near-lossless at the paper's quality-neutral point |
| FusedTQ3, K+V | **0.9538** | 0.5087 | Good average quality, occasional outlier positions |
| FusedTQ4, boundary V | **0.9825** | 0.8284 | First/last 2 layers at fp16 V |

> Run full WikiText-2 perplexity evaluation:
> `uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 4 --quality`

### Throughput

Measured decode throughput (tok/s) on Qwen3-8B, NVIDIA GB10:

| Context | FP16 | FusedTQ3 | FusedTQ4 |
|--------:|:----:|:--------:|:--------:|
| 4,096 | 3.9 | 3.7 | 2.8 |
| 8,192 | 2.5 | 2.3 | 1.7 |
| 16,384 | 1.4 | 1.3 | 0.9 |
| 32,768 | 0.7 | 0.6 | 0.6 |

> KV cache compression is primarily a **memory optimization** — the goal is fitting more context and larger batches, not faster single-sequence generation. FusedTQ3 runs within 5-15% of FP16 throughput. FusedTQ4 has higher overhead due to nibble packing/unpacking in the attention kernel.
>
> Reproduce: `uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --long-context`

### 🆚 vs TQ (Dejan.ai) / TQ+ (TheTom)

| Feature | FusedTQ (ours) | TQ (Dejan.ai) | TQ+ (TheTom) |
|---------|:-:|:-:|:-:|
| Compression target | **K + V (packed)** | K only | K + V |
| 4-bit compression ratio | **~3.8x** | ~1.3x | ~3.8x |
| 3-bit compression ratio | **~4.9x** | N/A | N/A |
| Kernel pipeline | **Fused 1-kernel (Triton)** | Multi-kernel (PyTorch) | Multi-kernel (C / Metal / CUDA) |
| Fused Q·K^T from packed | **Yes** | No | No |
| Rotation | **RHT O(d log d)** | Dense QR O(d²) | Walsh-Hadamard O(d log d) |
| Rotation storage/layer | **~1 KB** | ~256 KB | ~1 KB |
| Layer-aware V compression | **Yes (configurable callable)** | No | Yes (fixed boundary only) |
| HuggingFace integration | **Yes** | No | No (llama.cpp only) |
| QK norm support | **Yes** | No | N/A |

Full benchmark sweep: `uv run python benchmarks/run_fused_benchmark.py`

## 🏗️ Architecture

```
src/fused_turboquant/
├── core/
│   ├── hadamard.py         # RHT rotation (Triton primary, PyTorch fallback)
│   ├── lloyd_max.py        # Lloyd-Max quantizer for Beta distribution
│   ├── quantizer.py        # TurboQuantMSE: auto-selects fused/unfused
│   └── packing.py          # Sub-byte packing (4-bit: nibble, 3-bit: bitstream, 2-bit: 4/byte)
├── kernels/
│   ├── triton_rht.py       # Standalone RHT kernel
│   ├── triton_encode.py    # Fused encode: RHT + norm + quantize + pack
│   ├── triton_decode.py    # Fused decode: unpack + dequant + denorm + inv RHT
│   └── triton_attention.py # Fused Q·K^T from packed indices (inline nibble unpack)
├── hf/
│   └── fused_cache.py      # Compressed K+V storage + fused attention forward
├── vllm_plugin/            # Full vLLM attention backend (paged compressed KV cache)
└── cache/kv_cache.py       # Standalone KV cache wrapper
```

## 📋 Compatibility

| Dependency | Minimum | Tested up to | Install extra |
|:----------:|:-------:|:------------:|:-------------:|
| Python | 3.10 | 3.12 | — |
| PyTorch | 2.4 | 2.11 | — |
| Triton | 3.0 | 3.6 | `[cuda]` |
| transformers | 4.45 | latest | `[hf]` |
| vLLM | 0.8 | 0.18 | `[vllm]` |

## 🧪 Development

```bash
git clone https://github.com/Argonaut790/fused-turboquant.git
cd fused-turboquant
pip install -e ".[dev]"

pytest                                           # unit tests
python benchmarks/run_fused_benchmark.py         # kernel microbenchmarks
python benchmarks/run_full_comparison.py         # quality + memory benchmarks
```

**Real model benchmarks** (requires `pip install -e ".[dev,hf]"`):

```bash
# 3-way comparison: FP16 vs FusedTQ vs TQ (Dejan.ai)
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --max-new-tokens 512

# Long-context decode sweep (4K-32K tokens)
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --long-context

# Batch throughput search (find max batch per method)
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --batch-search

# Include WikiText-2 perplexity
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --quality

# Export results to JSON
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --json results.json
```

## 📝 Citation

This is an **independent, community-driven implementation** based on the TurboQuant algorithm described in [arXiv:2504.19874](https://arxiv.org/abs/2504.19874). It is **not affiliated with, endorsed by, or derived from code by** Google Research, Google DeepMind, or the original paper authors. The paper is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Our contribution is the fused Triton kernel design using RHT (replacing dense QR rotation) and the integrations with HuggingFace / vLLM — the implementation is entirely original.

If you use this implementation in your work, please cite both the original paper and this project:

```bibtex
@software{fused_turboquant,
  title   = {fused-turboquant: Fused Triton Kernels for TurboQuant KV Cache Compression},
  author  = {fused-turboquant Contributors},
  url     = {https://github.com/Argonaut790/fused-turboquant},
  year    = {2025},
  license = {Apache-2.0},
}

@inproceedings{zandieh2026turboquant,
  title     = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author    = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2504.19874},
}
```

## 🙏 Acknowledgements

- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni. ICLR 2026. The algorithm this project implements.
- [Fast JL Transform (RHT)](https://doi.org/10.1137/060673096) — Ailon & Chazelle. SICOMP 39(1), 2009. The rotation that enables kernel fusion.
- [Dejan.ai TurboQuant](https://dejan.ai/blog/turboquant/) — Dense QR implementation, benchmarked against.
- [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## 📄 License

Apache 2.0
