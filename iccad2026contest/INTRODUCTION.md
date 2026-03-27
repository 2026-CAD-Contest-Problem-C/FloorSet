# ICCAD 2026 FloorSet Challenge — 題目與 Baseline 介紹

## 一、競賽題目

### 背景

FloorSet 是由 Intel Labs 開發的 VLSI 晶片佈局規劃（Floorplanning）資料集，包含 200 萬個合成電路佈局。本競賽（ICCAD 2026 CAD Contest Problem C）要求參賽者實作一個佈局規劃演算法，對 100 個測試案例進行最佳化，以最低成本分數排名。

### 任務定義

實作 `solve()` 函式，輸入電路約束，輸出每個 block 的位置與尺寸：

```python
def solve(self, block_count, area_targets, b2b_connectivity,
          p2b_connectivity, pins_pos, constraints):
    # 回傳：List of (x, y, width, height)，每個 block 一筆
```

**輸入參數：**

| 參數 | 維度 | 說明 |
|------|------|------|
| `block_count` | int | block 數量（21～120） |
| `area_targets` | `[n]` | 每個 block 的目標面積 |
| `b2b_connectivity` | `[edges, 3]` | block 間連線：(block_i, block_j, weight) |
| `p2b_connectivity` | `[edges, 3]` | pin 到 block 連線：(pin_idx, block_idx, weight) |
| `pins_pos` | `[n_pins, 2]` | 固定 pin 的座標 (x, y) |
| `constraints` | `[n, 5]` | 每個 block 的約束：(fixed, preplaced, MIB, cluster, boundary) |

**約束說明：**

- **Fixed**：形狀須符合目標多邊形ji3，但可平移或旋轉
- **Preplaced**：形狀與位置均須固定
- **MIB（Multi-Instantiation Block）**：同 group ID 的 block 須共用相同形狀
- **Cluster**：同 group ID 的 block 的多邊形聯集須為連通多邊形
- **Boundary**：block 須靠齊指定邊界（LEFT=1, RIGHT=2, TOP=4, BOTTOM=8，可組合）

### 硬約束（違反 → 直接判為 infeasible，成本 = 10.0）

- Block 之間**不得重疊**
- 每個 block 的面積（w × h）必須在目標值的 **±1%** 以內

### 本競賽放寬的約束

| 約束 | 狀態 | 說明 |
|------|------|------|
| 長寬比 | 放寬 | 任意 width/height 比例均可 |
| Fixed Outline | 移除 | 改由 pin-to-block HPWL 和面積成本隱性處理 |
| 座標精度 | 放寬 | 允許浮點數座標 |

### 評分公式

```
Cost = (1 + 0.5 × (HPWL_gap + Area_gap)) × exp(2 × Violations) × RuntimeFactor
     = 10.0  （若 infeasible）
```

- **分數越低越好**，最終排名以 100 個測試案例的加權平均分數決定
- `HPWL_gap`：與 ground truth baseline 的 wirelength 差距比例
- `Area_gap`：與 ground truth baseline 的面積差距比例
- `Violations`：軟約束違反程度

### 資料集

| 資料集 | 數量 | 用途 | 是否公開 |
|--------|------|------|----------|
| Training | 100 萬 | 訓練 ML 模型 | 是（Hugging Face） |
| Validation | 100 | 本地調校與評估 | 是 |
| Test | 100 | 最終排名 | 隱藏（格式同 Validation） |

---

## 二、Baseline 演算法：B\*-tree Simulated Annealing

### 概覽

Baseline 結合兩個核心技術：**B\*-tree**（保證無重疊的佈局表示法）與 **Simulated Annealing**（全域最佳化搜尋策略）。

---

### 2.1 B\*-tree 資料結構

B\*-tree 是一種以二元樹表示晶片佈局的資料結構，透過樹的拓樸結構直接定義 block 之間的相對位置關係，**保證展開後不會產生重疊**。

**位置語義：**
```
左子節點 → 擺在父節點的「右邊」
右子節點 → 擺在父節點的「正上方」（相同 x 起點）
```

**展開（Pack）流程：**

`pack()` 使用 DFS 遍歷樹，搭配**天際線輪廓（Skyline Contour）**追蹤已放置區域的最高點：

```
contour = 目前已放置 block 形成的天際線（各 x 區間對應的最高 y）

放置新 block 時：
  1. 查詢目標 x 範圍的最高 y
  2. 將 block 底部對齊該 y
  3. 更新天際線
```

這個機制在數學上確保不重疊。

**樹的移動操作（SA 擾動用）：**

| 操作 | 說明 |
|------|------|
| `move_rotate(block)` | 旋轉 block（交換 w/h，面積不變） |
| `move_delete_insert(block)` | 從樹中移除後隨機插入至其他位置 |
| `move_swap(b1, b2)` | 交換兩 block 的尺寸（Baseline 未使用，以保持面積約束） |

---

### 2.2 Simulated Annealing

模擬金屬退火的物理過程：溫度高時允許接受較差解以跳出局部最佳，溫度逐漸降低後趨於收斂至最佳解。

**SA 參數：**

| 參數 | 值 |
|------|----|
| 初始溫度 | 100.0 |
| 終止溫度 | 1.0 |
| 冷卻率 | 0.9（每輪乘以 0.9） |
| 每溫度移動次數 | 20 |

**Metropolis 接受準則：**

```python
delta = new_cost - current_cost
if delta < 0:                          # 新解更好 → 直接接受
    accept
elif random() < exp(-delta / temp):    # 新解較差 → 以機率接受（溫度越高機率越大）
    accept
else:
    reject  # 還原至前一個狀態
```

---

### 2.3 成本函式

```python
cost = HPWL_b2b + HPWL_p2b + bbox_area × 0.01
```

| 項目 | 說明 |
|------|------|
| `HPWL_b2b` | Block 間半周長 wirelength（所有 b2b net 的加權 HPWL 總和） |
| `HPWL_p2b` | 固定 pin 到 block 中心的加權 HPWL 總和 |
| `bbox_area × 0.01` | 所有 block 的 bounding box 總面積（鼓勵緊湊排列） |

---

### 2.4 完整流程

```
初始化：所有 block 設為正方形（w = h = √area_target）
        ↓
建立隨機 B*-tree → pack() 計算初始座標 → 計算初始成本
        ↓
while temp > final_temp (1.0):
    重複 20 次：
        隨機選擇 rotate 或 delete-insert
        pack() 重新計算座標
        計算新成本
        依 Metropolis 準則接受或還原
        若為歷史最佳 → 記錄下來
    temp × 0.9（降溫）
        ↓
回傳歷史最佳佈局
```

---

### 2.5 Baseline 的限制

Baseline **不處理**下列約束，參賽者可從此處著手改進：

- Fixed block（固定形狀）
- Preplaced block（固定形狀與位置）
- MIB（多實例化 block 須共用形狀）
- Cluster（block 群須連通）
- Boundary（block 須靠邊）

---

## 三、快速開始

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 在 my_optimizer.py 中修改 solve() 方法

# 3. Install dependencies
pip install -r iccad2026contest/requirements.txt

# 對 validation set 評分
python iccad2026contest/iccad2026_evaluate.py --evaluate iccad2026contest/my_optimizer.py
```
