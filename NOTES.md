# 🐍 Snake RL 專案筆記 — 報告參考用

> 整理自開發過程中的對話，涵蓋助教回饋分析、演算法設計決策、比較重點。

---

## 1. 助教回饋 & 我們的對應

助教給了三段建議，以下是我們如何回應每一點：

### (1) "Highly feasible; manageable state/action space"
- 我們用 **11 維布林向量** 作為狀態表示，而非像素畫面
  - 3 個危險偵測（前方/右方/左方是否有牆或身體）
  - 4 個方向（目前蛇朝哪走）
  - 4 個食物相對位置（食物在蛇的上/下/左/右）
- 總狀態空間 = $2^{11} = 2048$，對表格法完美適用
- 動作空間 = 3（直走 / 右轉 / 左轉）

### (2) "Can easily implement custom version. Reward shaping is straightforward"
- **自訂 Gymnasium 環境** (`src/snake_env.py`)，非使用現成套件
- **Reward 設計**：
  - 基本：吃到食物 +10，死掉 -10，其他 0
  - Reward Shaping（`--reward_shaping`）：靠近食物 +0.1，遠離食物 -0.1（Manhattan distance）

### (3) "Consider comparing multiple algorithms to add depth"
- 實作了 **4 種演算法**（見下方）

---

## 2. 四種演算法總覽

| 演算法 | 類型 | 檔案 | 更新方式 | 訓練速度 |
|--------|------|------|----------|---------|
| **Tabular Q-Learning** | Value-based, Off-policy | `src/tabular_q.py` | 用「最佳」下一步更新 | ⚡ 秒級 |
| **SARSA** | Value-based, On-policy | `src/tabular_sarsa.py` | 用「實際走的」下一步更新 | ⚡ 秒級 |
| **DQN** | Value-based, Off-policy | `src/train_dqn.py` | Replay buffer + 神經網路 | 🐢 分鐘級 |
| **PPO** | Policy Gradient, On-policy | `src/train_ppo.py` | Clipped surrogate + Actor-Critic | 🐢 分鐘級 |

---

## 3. 演算法比較重點（報告核心）

### 3.1 Q-Learning vs SARSA（Off-policy vs On-policy）
- **Q-Learning** 更「大膽」：假設未來永遠選最好的動作來更新 → 蛇偏向冒險，敢貼牆走
- **SARSA** 更「保守」：用實際（含探索）動作來更新 → 蛇會刻意遠離危險，寧可繞路
- **報告亮點**：off-policy vs on-policy 導致的 **風險偏好差異 (risk-sensitivity)**

### 3.2 表格法 vs 深度學習（Tabular vs Neural Network）
- **表格法**：2048 個狀態可以精確記錄 → 訓練充足後 **表現穩定**，但無泛化能力
- **DQN**：Neural Network 逼近 Q 值，有泛化能力但 **學習曲線波動大**（overestimation bias）
- **PPO**：直接學 policy 而非 Q 值，收斂更穩定但在小狀態空間可能 **不如表格法**
- **報告亮點**：小狀態空間中 **表格法可能勝過深度學習**（殺雞用牛刀現象）

### 3.3 Reward Shaping 消融實驗
- 預期結果：對表格法影響大（加速收斂），對 DQN/PPO 的影響較小
- **報告亮點**：dense vs sparse reward 對不同演算法的影響差異

---

## 4. 超參數搜索

PDF Proposal 提到需要做 hyperparameter sweep，我們搜索的參數：
- **學習率 α**：`[0.01, 0.05, 0.1, 0.2]`
- **折扣因子 γ**：`[0.9, 0.95, 0.99]`
- **ε-decay**：`[0.99, 0.995, 0.999]`

產出：
- `results/hyperparameter_sweep_results.csv` — 完整表格
- `results/figures/hyperparameter_heatmap.png` — 熱力圖（α vs γ）

---

## 5. 如何跑實驗

```bash
# 啟動環境
source venv/bin/activate

# -- 個別訓練 --
python src/tabular_q.py --episodes 5000
python src/tabular_sarsa.py --episodes 5000
python src/train_dqn.py --timesteps 300000
python src/train_ppo.py --timesteps 300000

# -- 一鍵跑完所有比較 + 產圖 --
python src/compare_algorithms.py
# → 輸出 results/figures/algorithm_comparison.png（4 面板圖）
# → 終端輸出統計摘要表

# -- 超參數搜索 --
python src/hyperparameter_sweep.py --episodes 1000

# -- 看蛇玩遊戲 --
python src/evaluate.py --algo tabular
python src/evaluate.py --algo sarsa
python src/evaluate.py --algo dqn
python src/evaluate.py --algo ppo

# -- TensorBoard --
tensorboard --logdir logs/dqn_tensorboard/
tensorboard --logdir logs/ppo_tensorboard/
```

---

## 6. `compare_algorithms.py` 產出的 4 面板圖

| 面板 | 內容 | 報告用途 |
|------|------|---------|
| (a) Learning Curves | 所有演算法的分數 vs episode 曲線 | 比較收斂速度 |
| (b) Bar Chart | 最終平均分數長條圖 | 比較最終表現 |
| (c) Reward Shaping Ablation | 有/無 reward shaping 的 Q-Learning 對比 | 獎勵設計討論 |
| (d) Box Plot | 最後 200 episodes 的分數分佈 | 穩定性分析 |

---

## 7. 報告建議結構

1. **Introduction** — 問題描述、Snake 遊戲規則
2. **Environment Design** — 11 維狀態空間設計、動作空間、獎勵函數
3. **Algorithms** — 四種演算法介紹 + 數學公式
4. **Experiments**
   - 超參數搜索結果（heatmap）
   - 演算法比較（4 面板圖 + 統計表）
   - Reward Shaping 消融實驗
5. **Discussion** — Off-policy vs On-policy、表格法 vs 深度學習、挑戰與限制
6. **Conclusion** — 最佳演算法推薦、未來工作（更大的 grid、CNN-based DQN 等）

---

## 8. 專案檔案結構

```
project/
├── src/                                # 所有原始碼
│   ├── _paths.py                       # 共用路徑常數
│   ├── snake_env.py                    # 自訂 Gymnasium 貪吃蛇環境
│   ├── tabular_q.py                    # Tabular Q-Learning
│   ├── tabular_sarsa.py                # SARSA
│   ├── train_dqn.py                    # DQN (Stable-Baselines3)
│   ├── train_ppo.py                    # PPO (Stable-Baselines3)
│   ├── compare_algorithms.py           # 全演算法比較 + 4 面板圖
│   ├── evaluate.py                     # Pygame 視覺化評估
│   └── hyperparameter_sweep.py         # 超參數搜索 + 熱力圖
├── results/
│   ├── models/                         # 訓練好的模型 (.pkl, .zip)
│   └── figures/                        # 產生的圖表 (.png)
├── logs/                               # TensorBoard & 訓練日誌
├── requirements.txt
├── README.md
└── NOTES.md                            ← 你正在看的這份

---

## 9. Dataset Requirement (Project with a Dataset)

為了符合專案要求中 "**moderately large-sized data set**" 的評分標準，我們在 Notebook 中加入了 **離線模仿學習 (Offline Imitation Learning / Behavior Cloning)** 的設計：

1. **專家資料集產生 (Expert Dataset)**：我們讓訓練好的 Q-Learning 代理人玩 1,000 場遊戲，收集了幾十萬筆的 `(狀態, 動作)` 紀錄，這就是我們的 dataset。
2. **模仿學習演算法 (Behavior Cloning)**：這個演算法不玩遊戲，純粹從 Offline Dataset 中統計機率來複製專家的行為。
3. **針對評分標準的回應**：
   - **Training Time**: Tabular 幾秒，Deep RL 幾分鐘，BC(Dataset) 只要零點幾秒（不用玩環境）。
   - **Results vs Expected**: 解釋為什麼模仿學習表現不佳（因為發生 Covariate Shift）。
   - **Important parameters**: 提到 Gamma($\gamma$), Epsilon($\epsilon$) 還有 Dataset Coverage 的重要性。
   - **Challenges**: 分析了 Online RL 的稀疏獎勵挑戰，以及 Dataset Learning 的痛點 Covariate Shift。
   - **Future Work**: 提到 DAgger 演算法和 CNN 升級版。
