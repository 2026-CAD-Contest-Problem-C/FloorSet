# FloorSet GNN + RL Floorplanning Agent

ICCAD 2026 FloorSet Challenge — Phase 1 Implementation

## 系統架構

```
FloorSet sample
    └─► Data Pipeline (graph_builder + feature_extractor)
            └─► GNN Encoder (3-layer GIN)
                    ├─► Shape Policy (MLP → log aspect ratio → w, h)
                    └─► Placement Policy (Actor-Critic + PPO)
                                └─► Canvas (prefix-sum legality check)
                                        └─► Reward (HPWL gap + area gap)
```

### 各元件說明

| 元件 | 檔案 | 說明 |
|------|------|------|
| Graph Builder | `floorplan/data/graph_builder.py` | FloorSet sample → PyG Data（含 Laplacian PE） |
| GNN Encoder | `floorplan/models/gnn_encoder.py` | 3-layer GIN + JK concat + residuals → 128-dim embedding |
| Shape Policy | `floorplan/models/shape_policy.py` | MLP → log_r → (w, h)，維持面積約束 |
| Placement Policy | `floorplan/models/placement_policy.py` | Actor-Critic，ConvTranspose2d 空間解碼 |
| Canvas | `floorplan/env/canvas.py` | 2D prefix sum O(1) legality check，保證零重疊 |
| Env | `floorplan/env/floorplan_env.py` | Gym-compatible 環境，依 b2b degree 排序放置順序 |

---

## 資料路徑

```
訓練資料 (1M samples)：/home/syhuang/floorset_lite/
    └─ worker_{0..99}/layouts_*.th   (每個 worker 90 個檔，每檔 112 個 layout)

測試資料 (100 samples)：/home/syhuang/FloorSet/LiteTensorDataTest/
    └─ config_{n}/litedata_1.pth
               /litelabel_1.pth

Checkpoint 輸出：./checkpoints/
Cache 目錄：/home/syhuang/FloorSet/iccad2026contest/solution/cache/
```

設定檔：`floorplan/configs/default.yaml`

---

## 安裝依賴

```bash
pip install torch>=2.0.0 torch-geometric>=2.4.0 gymnasium scipy wandb tqdm pyyaml
```

---

## 訓練流程

所有指令在 `iccad2026contest/solution/` 目錄下執行。

### Step 0：執行單元測試（建議先跑，確認環境正常）

```bash
python3 -m pytest iccad2026contest/solution/floorplan/tests/ -v
```

預期結果：22 passed

---

### Step 1：GNN 自監督預訓練（Phase 1a）

同時執行三個自監督任務，讓 GNN 在沒有任何實際擺放結果的情況下，預先理解 floorplan 問題的結構：

| Task | Weight | 方法 | 希望模型學到的能力 |
|------|--------|------|-----------------|
| **A** | 1.0 | 遮蔽 15% 節點特徵，用鄰居 embedding 還原（MSE） | 從圖的結構和鄰居推斷節點屬性（面積、constraint）；理解「哪種位置的 block 通常長什麼樣」 |
| **B** | 0.5 | 預測兩節點之間是否有邊（Binary Cross-Entropy） | 理解 netlist connectivity；讓有邊的節點在 embedding 空間距離近，沒有邊的距離遠 |
| **C** | 1.0 | 從 graph embedding 預測 log(HPWL+1)（MSE） | 從整張圖的拓樸結構評估布線複雜度；讓 graph embedding 帶有「這個 floorplan 有多難布線」的全域語義 |

三個任務的分工：A 學節點局部語義、B 學節點間結構關係、C 學全域品質評估，共同讓 GNN 具備足夠的表徵能力，使後續 RL 訓練更快收斂。

**Loss 組合**：`loss = 1.0 * A + 0.5 * B + 1.0 * C`（以 Task C val HPWL MSE 做 early stopping）

```bash
python main.py pretrain \
    --config floorplan/configs/default.yaml \
    --output checkpoints
```

**預期時間**：4–8 小時（單 GPU，50 epochs，1M samples）

**Curriculum 排程**：
- Epoch 1–10：只用 21–40 blocks（smoke test 用小圖）
- Epoch 11+：全部大小

**Early stopping**：val Task-C HPWL MSE 5 epochs 不進步即停止

**輸出**：`checkpoints/gnn_pretrained.pt`

**煙霧測試（只跑 1 epoch，確認流程通順）**：

```bash
python3 -c "
import yaml
with open('floorplan/configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['gnn']['pretrain_epochs'] = 1
cfg['gnn']['pretrain_batch'] = 8
import sys; sys.path.insert(0, '.')
from floorplan.training.pretrain_gnn import pretrain
pretrain(cfg, output_dir='checkpoints')
"
```

---

### Step 2：PPO 強化學習訓練（Phase 1b）

以預訓練 GNN 為起點，用 PPO 訓練 Placement Policy。

```bash
python main.py train \
    --config floorplan/configs/default.yaml \
    --gnn_ckpt checkpoints/gnn_pretrained.pt \
    --output checkpoints
```

**Curriculum 四階段**：

| 階段 | Block 數量 | 目標 |
|------|-----------|------|
| Stage 1 | 21–40 | feasibility rate = 100% |
| Stage 2 | 41–70 | 提升 HPWL/area 品質 |
| Stage 3 | 71–100 | 泛化到中大型 |
| Stage 4 | 101–120 | 最大規模 |

每階段保留 20% batch 來自前期資料（experience replay），防止遺忘。

**GNN 凍結**：前 5,000 steps 凍結 GNN weights，先讓 placement policy 穩定。

**輸出**：
- `checkpoints/best_agent.pt`（val reward 最佳）
- `checkpoints/agent_stage{1..4}.pt`（各階段最終）

---

### Step 3：視覺化比較（Agent vs GT）

比較 agent 擺放結果與 GT，確認訓練品質：

```bash
# 用 config 編號指定（推薦）
python visualize_compare.py \
    --ckpt checkpoints/best_agent.pt \
    --config_id 34 \
    --save compare_config34.png

# 批次比較多個 config
for i in 34 71 95 100 120; do
  python visualize_compare.py \
      --ckpt checkpoints/best_agent.pt \
      --config_id $i \
      --save compare_config${i}.png
done

# 也可用 0-based 索引（sorted 順序）
python visualize_compare.py \
    --ckpt checkpoints/best_agent.pt \
    --case 0 \
    --save compare_case0.png
```

**輸出圖示說明**：

| 顏色 | 意義 |
|------|------|
| Steel blue | 一般 block |
| Violet | Fixed position |
| Gray | Preplaced |
| Dark green | Must-in-block |
| Salmon | Cluster |
| Olive | Boundary |
| 紅線 | Block-to-block 連線（b2b） |
| 藍線 | Pin-to-block 連線（p2b） |
| 綠點 | Pin 位置 |

左圖為 Agent 結果，右圖為 GT，並顯示各自 HPWL 供比較。

---

### Step 4：評估（100 個測試案例）

```bash
python main.py evaluate \
    --config floorplan/configs/default.yaml \
    --ckpt checkpoints/best_agent.pt \
    --output solutions
```

**輸出**：
- `solutions/evaluation_results.json`：每個 case 的詳細指標
- `solutions/solutions.json`：所有 case 的 placement 結果

**關鍵指標**：

| 指標 | 目標 |
|------|------|
| Feasibility rate | 1.000（必須 100%，無 overlap） |
| Mean HPWL gap | 越低越好（< 0.5 為佳） |
| Mean area gap | 越低越好（< 0.5 為佳） |
| Mean cost | 越低越好（< 2.0 為佳） |

**注意**：提交前必須確認 feasibility rate = 1.000，否則 cost = 10.0。

---

## Reward 設計

```
overlap > 0   →  reward = -10.0  (hard penalty)
overlap = 0   →  cost = 1.0 + 0.5 * (hpwl_gap + area_gap)
               reward = -cost
```

Moving average baseline（α=0.99）減少 variance：`advantage = reward - baseline`

---

## 重要超參數（`floorplan/configs/default.yaml`）

```yaml
gnn:
  hidden_dim: 128        # node/graph embedding 維度
  num_layers: 3          # GIN layer 數
  pretrain_epochs: 50    # 預訓練總 epoch 數
  pretrain_batch: 32     # 預訓練 batch size

placement:
  grid_size: 256         # 離散化 canvas 解析度
  ppo_lr_actor: 3e-4
  ppo_clip: 0.2
  freeze_gnn_steps: 5000 # RL 開始時凍結 GNN 的 steps 數

curriculum:
  stages: [[21,40],[41,70],[71,100],[101,120]]
  replay_frac: 0.2       # 前期資料 replay 比例
```

---

## 目錄結構

```
solution/
├── main.py                        # 統一入口點
├── visualize_compare.py           # Agent vs GT 視覺化比較
├── floorplan/
│   ├── configs/default.yaml       # 所有超參數
│   ├── data/
│   │   ├── graph_builder.py       # sample → PyG Data
│   │   ├── feature_extractor.py   # Laplacian PE + 12-dim 節點特徵
│   │   └── dataset.py             # 懶加載 + disk cache + curriculum
│   ├── models/
│   │   ├── gnn_encoder.py         # 3-layer GIN
│   │   ├── shape_policy.py        # log aspect ratio → (w, h)
│   │   ├── placement_policy.py    # Actor-Critic
│   │   └── full_agent.py          # 整合三個模型
│   ├── env/
│   │   ├── canvas.py              # prefix-sum legality check
│   │   └── floorplan_env.py       # Gym 環境
│   ├── training/
│   │   ├── losses.py              # 預訓練 loss + PPO + reward
│   │   ├── metrics.py             # HPWL / area / overlap / cost
│   │   ├── pretrain_gnn.py        # Phase 1a
│   │   └── train_rl.py            # Phase 1b PPO
│   ├── evaluation/
│   │   ├── evaluate.py            # 評估 100 test cases
│   │   └── submit.py              # 格式化提交
│   └── tests/
│       ├── test_canvas.py         # 9 tests
│       ├── test_gnn.py            # 8 tests
│       └── test_reward.py         # 5 tests
└── checkpoints/                   # 訓練產出（自動建立）
```
