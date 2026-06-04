# vLLM バックエンド 詳細仕様

`fused-turboquant` の vLLM プラグイン (`attention_backend="TURBOQUANT"`) に関する設定キー、 性能モード、 K / V 独立 bit 幅、 境界層保護の bit 幅、 スロットレイアウトをまとめた詳細リファレンスです。 まず動かしたい場合は [README の Quick Start](../README.md#-vllm) を参照してください。

## `additional_config["turboquant"]` 全 key リファレンス

`LLM(..., additional_config={"turboquant": {...}})` または `vllm serve --additional-config '{"turboquant": {...}}'` で渡せる全 key:

| key | 型 | デフォルト | 説明 |
|---|---|---|---|
| `kind` | str | `"rht"` | 回転種別: `rht` / `rotor` / `planar` / `iso_fast` / `iso_full` |
| `boundary_protect` | int / str | `1` | 境界 4 層の扱い: `0` / `off` (無効)、 `1` / `fp16` (FP16 raw)、 `fp8` (FP8 K)、 `2`〜`8` (MSE quantization) |
| `v_rotate` | bool | `False` | V を K と同じ rotation で pre-rotate + post-stage2 invert |
| `v_lloyd_max` | bool | `False` | V を Lloyd-Max codebook + per-vec norm で量子化 (実モデルでは uniform より劣る場合あり、 V_ROTATE と併用) |
| `key_bits` / `value_bits` | int | preset の bit 幅 | preset 内で K / V の実効 bit を独立に override (1〜4) |
| `defer_prefill` | bool | `False` | prefill 中は K を FP16 で保持、 decode 時に量子化 (長 context 向け、 BP=1 必須) |
| `prefill_fa_version` | int | (vLLM 設定) | prefill の FlashAttention バージョン (2/3/4)。 SM 12 で FA4 を試すと runtime で FA2 に自動 fallback |
| `cudagraph_mode` | str | (vLLM 既定) | CUDA graph モード (`FULL` / `PIECEWISE` / `FULL_AND_PIECEWISE` / `NONE`) — 通常は触らない |
| `boundary_fa_paged` | bool | `False` | boundary 層を paged Flash Attention に戻す (Blackwell の non-contig stride バグの絡みで default は自作 Triton kernel) |

各 key は対応する環境変数 (`TURBOQUANT_KIND` / `TURBOQUANT_BOUNDARY_PROTECT` …) にも `os.environ.setdefault` 経由で転記されます。 つまり明示的に env var を設定してある場合はそちらが優先 (後方互換)。

`kv_cache_dtype` は未指定で OK: `attention_backend="TURBOQUANT"` か `additional_config["turboquant"]["key_bits"]` / `value_bits` が指定されていれば、 plugin が最小フィット host preset を自動で埋めます (`K, V ≤ 3` → `turboquant_3bit_nc` / `V = 4` → `turboquant_k3v4_nc` / それ以外 → `turboquant_4bit_nc`)。 明示的に渡した場合はそちらが優先。

備考:
- **`BP=0` は `kind="rht"` 限定で安全**。 block-diag 系 (planar / rotor / iso_fast / iso_full) は boundary 層を gaussian 化しきれないので `boundary_protect="fp8"` か `1` を併用してください。
- **`v_rotate=True` は全 quant 構成で精度に効きます** (Gemma 4 31B + rotor で +4.3〜+72 pts の改善)。 計算オーバーヘッドは per-decode-step の `output @ M^T` GEMM 1 回 (≪ 0.1 ms)。
- **`prefill_fa_version`** は `AttentionConfig(flash_attn_version=4)` (CLI: `--attention-config.flash_attn_version=4`) でも指定可能。 vLLM stock は TurboQuant 使用時に強制 FA2 化するが、 本 plugin が `additional_config["turboquant"]["prefill_fa_version"]` / `attention_config.flash_attn_version` を内部の `TQ_PREFILL_FA_VERSION` に転記して上書きするため、 FA3 / FA4 が使える GPU では prefill だけ高速化できます。 SM 12 (RTX PRO 6000 Blackwell) では vLLM 0.20.0 の FA4 backend (`FlashAttentionForwardSm120`) に未修正バグがあり runtime で FA2 に fallback します (ログ 1 回)。 Hopper SM 9 や B100 / B200 では FA3 / FA4 が実効的に有効。

## 性能モード一覧

vLLM バックエンドには 4 つの独立した切り替えスイッチがあり、 これらを組み合わせることで decode 速度・圧縮率・対応 prefix 長を調整できます。

| モード | 環境変数 / フラグ | 効果 | 推奨用途 |
|---|---|---|---|
| **CUDA graphs** | `enforce_eager=False` *(vLLM)* | decode 1 ステップごとの kernel 起動 overhead を償却 — 小型モデルで **decode 5〜14 倍**、 30B+ で約 1.7 倍 | 本番では **常に ON** |
| **Boundary protection (境界層保護)** | `TURBOQUANT_BOUNDARY_PROTECT=1\|fp16\|2\|3\|4\|8` | 最初と最後の 2 層を **指定 bit 数で** 保持。 `1` / `fp16` (デフォルト) = FP16 raw; `2`〜`4` / `8` = TQ 量子化の bit 数を上書き (slot は FP16 サイズ確保のままで余裕あり)。 精度向上と同時に、 上位 bit (FP16 / 8-bit) の層が centroid lookup より速く attention できるため throughput も BP=0 を上回ることが多い。 `defer_prefill` を CUDA graph で正しく動かすために **必須** | 圧縮率を限界まで詰めたい場合を除き **常に ON** |
| **Deferred FP16 K-cache (遅延 K キャッシュ量子化)** | `TURBOQUANT_DEFER_PREFILL=1` | prefill トークンの回転・量子化をスキップし、 サイドバッファに FP16 のまま保持。 decode 時に 2 段階 attention (FP16 prefix 上で flash-attn + 量子化領域上で offset-decode kernel、 log-sum-exp で merge) を実行 | **長コンテキスト (prefix ≥ 4〜5K tokens) のみ**。 それ未満では遅くなる (速度表参照) |
| **回転種別** | `TURBOQUANT_KIND=rht\|planar\|rotor\|iso_fast\|iso_full` | MSE 量子化の前に適用する直交変換を選択。 RHT (Walsh-Hadamard butterfly) が本番標準、 Planar / Rotor / Iso はブロック対角型 (2 x 2 / 3 x 3 / 4 x 4) で各々 in-kernel 回転実装あり | 精度重視は RHT、 rotorquant 論文派生は Planar / Iso |

### ワークロード別 推奨設定

| ワークロード | `enforce_eager` | `TURBOQUANT_BOUNDARY_PROTECT` | `TURBOQUANT_DEFER_PREFILL` |
|---|---|---|---|
| 短い chat (prefix ≤ 4K tokens) — **既定** | `False` | `1` | `0` |
| 長コンテキスト RAG / agent (prefix ≥ 5K tokens) | `False` | `1` | `1` |
| 圧縮率優先 / decode が遅くてもよい場合 | `False` | `0` | `0` |

### 実測 decode 速度 (Qwen 2.5-3B-Instruct, RTX PRO 6000, planar)

各設定を切り替えたときの decode スループットを示します。 プロンプト長は **入力トークン数 (tokens)**、 表中の数値は **decode ステップあたりのトークン生成速度 (tok/s)** です。 **太字** は各行の最速設定。

| プロンプト長 [tokens] | Eager + BP=0 [tok/s] | + CUDA graphs [tok/s] | + BP=1 [tok/s] | + defer (BP=1) [tok/s] |
|---:|---:|---:|---:|---:|
| 181 | 35.5 | 194.7 | **193.2** | 157.7 |
| 1,201 | 22.5 | 181.8 | **181.8** | 157.1 |
| 4,801 | 10.4 | 147.4 | 150.2 | **155.3** |
| 19,201 | 38.6 | 88.7 | 93.8 | **155.6** |

### カーネル単体の速度 (RHT vs 各ブロック対角型, RTX PRO 6000)

fused store カーネル単体 (rotation + bucketize + V quant + slot scatter まで含む 1 launch) を計測したものです。 `matrix(D, D)` は最初に実装した汎用 `(D, D)` 行列を SRAM タイルでまるごと matmul する経路で、 ブロック対角型の理論優位性を計測する比較基準として残しています (実運用では block-diag 戦略がデフォルト)。 1024 トークン x KV 8 ヘッド x バッチ 1。

| 設定 | RHT (FWHT butterfly) [µs] | matrix(D, D) [µs] | planar (B=2) [µs] | rotor (B=3) [µs] | iso_fast (B=4) [µs] | iso_full (B=4) [µs] |
|---|---:|---:|---:|---:|---:|---:|
| **prefill 1024 tok, D=128** | 100.3 | 114.4 | **100.2** | **99.9** | **99.5** | 100.3 |
| **prefill 1024 tok, D=256** | 106.6 | **3,088.3** | **100.0** | **100.4** | **100.2** | **99.8** |
| **decode 1 tok, D=128** | 100.7 | 88.9 | 100.4 | 100.1 | 100.4 | 100.7 |
| **decode 1 tok, D=256** | 99.6 | 92.6 | 101.8 | 101.0 | 100.8 | 100.1 |

ポイント:
- **D=256 prefill**: 汎用 `(D, D)` matmul カーネルは 65 KB の float32 タイルが SRAM に乗らずタイル化が走るため **約 30 倍遅く** なる。 これに対し block-diag (Planar / Rotor / Iso) はブロックあたり 4〜16 FMA で済むため約 100 µs に揃う。 これが [rotorquant の論文](https://www.scrya.com/rotorquant/) が主張する「TurboQuant 比 10〜19 倍速い」の正体で、 本実装でも同等の効果が得られています。
- **D=128 prefill**: `(D, D)` matmul も SRAM に乗るので差は小さい (約 14%)。 RHT / block-diag / matrix のいずれもほぼ同じ。
- **decode 1 token**: 8 ヘッド x 1 トークンとワークが小さく、 すべてのカーネルが約 100 µs で頭打ち = **CUDA launch overhead 律速**。 実プログラムを 1 launch 単位で見ると差はほぼ noise レベルなので、 end-to-end の decode スループットには直接効きません。
- block-diag 戦略間 (Planar / Rotor / Iso_Fast / Iso_Full) は B=2〜4 で FMA 数こそ違うものの、 measure 上は全て約 100 µs に揃う (B が小さいほど僅かに速いがやはり launch 律速)。 **精度の差はあっても速度の差は微小**、 というのが実装上の結論です。

## K と V に独立な bit 数を割り当てる

実効 K / V bit を制御する方法は 2 つあります:

1. **`additional_config["turboquant"]["key_bits"]` / `value_bits`** (推奨): `LLM(..., additional_config={"turboquant": {"key_bits": 3, "value_bits": 2, ...}})`。 `kv_cache_dtype` は未指定で OK (plugin が最小フィット host preset を自動補填)。
2. **`kv_cache_dtype="turboquant_k{K}v{V}_nc"`**: `K, V ∈ {1, 2, 3, 4}` の全 16 通りを直接プリセット名として指定可能。

```python
llm = LLM(
    model="...",
    attention_backend="TURBOQUANT",
    kv_cache_dtype="turboquant_k3v2_nc",   # K=3-bit, V=2-bit
)
```

```bash
vllm serve ... --kv-cache-dtype turboquant_k2v4_nc   # K=2-bit, V=4-bit
```

| K | V | プリセット名 |
|:---:|:---:|---|
| 4 | 4 | `turboquant_k4v4_nc` (= `turboquant_4bit_nc`) |
| 4 | 3 | `turboquant_k4v3_nc` |
| 4 | 2 | `turboquant_k4v2_nc` |
| 4 | 1 | `turboquant_k4v1_nc` |
| 3 | 4 | `turboquant_k3v4_nc` (= ストック preset と同名) |
| 3 | 3 | `turboquant_k3v3_nc` (= `turboquant_3bit_nc`) |
| 3 | 2 | `turboquant_k3v2_nc` |
| 3 | 1 | `turboquant_k3v1_nc` |
| 2 | 4 | `turboquant_k2v4_nc` |
| 2 | 3 | `turboquant_k2v3_nc` |
| 2 | 2 | `turboquant_k2v2_nc` |
| 2 | 1 | `turboquant_k2v1_nc` |
| 1 | 4 | `turboquant_k1v4_nc` |
| 1 | 3 | `turboquant_k1v3_nc` |
| 1 | 2 | `turboquant_k1v2_nc` |
| 1 | 1 | `turboquant_k1v1_nc` |

### 仕組み

1. vLLM stock の `kv_cache_dtype` は `Literal[...]` 型で 4 種のプリセットしか受け付けない。
2. `fused_turboquant.vllm_plugin` の `_patch_extend_tq_presets` が `EngineArgs.__post_init__` をフックして、 `turboquant_k{K}v{V}_nc` を最小の **ホストプリセット** に書き換え:
   - `K, V ≤ 3` → `turboquant_3bit_nc`
   - `K ≤ 3, V = 4` → `turboquant_k3v4_nc`
   - その他 (`K = 4` か `V > 3`) → `turboquant_4bit_nc`
3. 同時に `TURBOQUANT_KEY_BITS=K` / `TURBOQUANT_VALUE_BITS=V` を `os.environ.setdefault` で設定。
4. `v1_backend.py` の bit override path が environ を読んで、 量子化 kernel に実効 bit 幅を渡す。
5. `_slot_size_for` も同じ env var を参照し、 **実効 bit から最小スロットバイト数 `ceil(K·D/8) + 2 + ceil(V·D/8) + 4` を計算**して `get_kv_cache_shape` に返す。 これにより paged cache の物理メモリも実効 bit に合わせて縮みます (後述のとおり power-of-2 切り上げと boundary protection の影響あり)。

### スロットレイアウトと実効メモリ

slot サイズは実効 bit から逆算され (`_slot_size_for` が `ceil(K·D/8) + 2 (K norm) + ceil(V·D/8) + 4 (V scale + zero)` を計算 → power-of-2 切り上げ)、 paged cache の物理メモリに直接反映されます。 head_dim=128 での実効スロットバイト数:

| K + V | raw bytes | スロット [bytes] | FP16 (512 B / token) 比 圧縮率 |
|:---:|---:|---:|---:|
| ≤ 3 (e.g. K=V=1, K=1V=2) | ≤ 54 | **64** | **8.0 x** |
| 4 〜 7 (e.g. K=V=2, K=V=3, K=3V=4) | 70 〜 118 | **128** | **4.0 x** |
| 8 (K=V=4) | 134 | **256** | **2.0 x** |

従来は host preset (`turboquant_3bit_nc` / `turboquant_k3v4_nc` / `turboquant_4bit_nc`) の `slot_size_aligned` のみで slot が決まっていたため、 `key_bits=2, value_bits=2` を指定しても `3bit_nc` 由来の 102 → pow2 128 のままでした。 現在は実効 bit から再計算するので、 例えば K=V=1 にすると 64 bytes (8 x 圧縮) まで詰まります。

ただし **boundary protection が有効** (`boundary_protect != 0`) の場合は、 境界層が要求する FP16 サイズ slot (`4·D` bytes) に揃えるため `_slot_size_for` の戻り値が `max(tq_raw, 2·D)` 以上に押し戻され、 さらに page-size unification shim によって block_size 拡張で吸収されます。 つまり **bit 削減による memory 節約は `boundary_protect=0` (= rht 必須) の場合に効きます**。 BP 有効モード (`fp16` / `fp8` / `2`〜`8`) では依然として量子化誤差の品質チューニングとしてのみ作用します (slot は boundary 由来の FP16 サイズで確保され、 余白部分は inert)。

### 実装上の補足

- 1-bit / 2-bit K / V は本リポジトリの `_bucketize_pack_norm_v_store` (store kernel) と `_tq_decode_stage1_offset` (decode kernel fork) に専用 branch を追加して対応。 **vLLM の stock decode kernel (`_tq_decode_stage1`) は 3-bit / 4-bit のみ対応**のため、 K / V のいずれかが 1-bit / 2-bit の場合は自動的に本リポジトリの fork を経由します。
- Lloyd-Max centroids は vLLM の `get_centroids(d, bits)` をそのまま使用 (1-bit = 2 levels、 2-bit = 4 levels、 3-bit = 8 levels、 4-bit = 16 levels)。
- 1-bit V (`turboquant_k*v1_nc`) は uniform per-vector min/max の 2 levels となり、 GSM-8K のような構造化タスクでは accuracy が急落 (Gemma 4 31B + K=2, V=1 で 48%、 K=V=1 で 0%)。 実用下限は **K2V2** (95% accuracy, 14.6 x compression)。
- FP8 keys (`turboquant_k8v4`) は v1 backend では未サポート (boundary 層のみ `TURBOQUANT_BOUNDARY_PROTECT=fp8` で利用可能)。

## 境界層保護の bit 数を指定する (FP16 以外の量子化で境界層を保護)

境界層保護 (Boundary Protection) は従来「最初と最後の 2 層を FP16 で保持」する機能でしたが、 `TURBOQUANT_BOUNDARY_PROTECT` を bit 数で指定すると **境界層を任意の bit 数で量子化** できます。 FP16 (16-bit) より大幅にメモリ削減しつつ、 中間層 (3-bit / 4-bit) より高精度に保つ、 という中間の選択肢を得られます。

| `TURBOQUANT_BOUNDARY_PROTECT` | 境界層の扱い | スロット使用バイト数 [bytes] (head_dim=128 時) |
|---|---|---:|
| `0` / `off` | 保護なし (全層中間層と同じ bit 数) | (中間層と同じ) |
| `1` / `fp16` (既定) | 境界層を **FP16 raw** で保持。 flash-attn / SDPA 経由 | 512 (4·D = K_fp16 + V_fp16) |
| `2` | 境界層を **2-bit** TQ で保持 | 70 (32 K + 2 norm + 32 V + 4 V scale/zero) |
| `3` | 境界層を **3-bit** TQ で保持 | 102 (48 K + 2 + 48 V + 4) |
| `4` | 境界層を **4-bit** TQ で保持 | 134 (64 K + 2 + 64 V + 4) |
| `8` | 境界層を **8-bit** TQ (256 levels MSE K + 256 levels uniform V) で保持 | 262 (128 K + 2 + 128 V + 4) |

スロット自体は FP16-sized (= `4·head_dim` bytes) のまま確保されており、 上記いずれの bit 数でも余裕で fit します (`head_dim=128` で 512 bytes 確保、 最大の 8-bit 設定でも 262 bytes 使用)。 残りは未使用領域。

### 使用例

```bash
# 境界層を 8-bit TQ で保護 (中間層は 4-bit のまま)
TURBOQUANT_BOUNDARY_PROTECT=8 \
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --attention-backend TURBOQUANT \
  --kv-cache-dtype turboquant_4bit_nc
```

### 実装メモ

- 境界層は vLLM の `kv_cache_dtype_skip_layers` 機構によって `cache_dtype="auto"` (FP16 spec) で確保されます。 境界層保護を bit 数指定にした場合も、 **スロット自体は FP16-sized のまま**で、 TQ kernels が先頭の数十〜数百バイトのみを使用します。 残りは inert (読み書きされません)。
- 8-bit boundary は 256 Lloyd-Max centroids + 256 levels uniform V。 本リポジトリの fork 版 decode kernel に `VQB == 8` branch を追加して実装。 `MSE_BITS == 8` 側は既存 kernel の汎用 bit-shift formula で自動的にカバーされます。
- 2-bit / 3-bit / 4-bit boundary は既存の middle-layer 用の同 bit-width 経路を流用します。
- `TURBOQUANT_KEY_BITS` / `TURBOQUANT_VALUE_BITS` (中間層用) は境界層には作用しません。 境界層は `TURBOQUANT_BOUNDARY_PROTECT` の bit 数で決まります (K, V 同 bit のみ; K / V 個別指定は今後の課題)。

## その他の補足

- 比較対象: [rotorquant](https://github.com/scrya-com/rotorquant) は llama.cpp 上の Qwen 2.5-3B planar3 で 119 tok/s と報告。 本実装では同条件で 193 tok/s、 19K トークン prefix でも 155 tok/s を維持します。
- `TURBOQUANT_DEFER_PREFILL=1` を使うには `TURBOQUANT_BOUNDARY_PROTECT=1` が必要です。 BP=0 + defer + cudagraph はクラッシュせず動きますが、 capture 時のディスパッチが不一致のため出力が無音で壊れます。
- defer 経路は層ごとに `[max_model_len, num_kv_heads, head_size]` の FP16 サイドバッファを確保するため、 量子化 paged cache に加えてメモリを消費します。
- `head_size > 256` のモデル (例: Gemma 4 31B のグローバル層) では境界層が flash-attn ではなく CUDA graph 安全な SDPA gather に fallback します。 CUDA graphs 自体は動きますが、 eager 比の高速化倍率は ~1.7 倍 (5 倍 + ではなく) になります。
