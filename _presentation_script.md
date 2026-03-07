# Snake RL Project: Video Presentation Script (3-5 mins)

## 🕒 Timing Breakdown / 時間分配
- **[0:00 - 0:45]** Introduction & Environment (Overview & State Space)
- **[0:45 - 2:15]** Algorithm Approaches (Online RL vs Offline Imitation Learning, Dataset Size Ablation)
- **[2:15 - 3:00]** Live Visualization (Side-by-Side Play)
- **[3:00 - 4:00]** Results, Setbacks, & Discussion (4-Panel Figure, Covariate Shift, CPU vs GPU)
- **[4:00 - 4:30]** Conclusion & Future Work

---

## 🎬 Section 1: Introduction (0:00 - 0:45)
**🎥 Screen Action:** Start at the top of the Notebook: `1. Overview & Course Fit`. Scroll down slightly to `2. Environment Design`.

**🗣️ Speaker (English):**
"Hi everyone. We are Willie and Wei-In. Welcome to our project on solving the classic Snake game using Reinforcement Learning. Our goal was not just to train a single agent, but to rigorously compare **Tabular methods**, **Deep RL**, and pure **Offline Imitation Learning** using a moderately large dataset. We built a custom Gymnasium environment representing the state as an efficient 11-dimensional compact vector."

**🗣️ Speaker (中文參考):**
「大家好，我們是 Willie 和 Wei-In。歡迎來到我們用強化學習解貪食蛇的專案。我們的目標不只是訓練一隻貪食蛇，而是要嚴格比較 **Tabular方法**、**Deep RL**，以及使用大型資料集的 **離線模仿學習 (Behavior Cloning)**。我們自己寫了一個 Gym 的環境，並且非常有技巧地把環境狀態濃縮成只有 11 個特徵的向量。」

---

## 🎬 Section 2: Online vs Offline Approaches (0:45 - 2:00)

**🎥 Screen Action:** Scroll down to `3. Algorithm Implementations` and pause at the **Q-Learning** (CELL 8) and **SARSA** (CELL 10) classes.

**🗣️ Speaker (English):**
"Moving to our implementations. First, we built pure Python classes for our tabular baselines. Here you can see **TabularQ** and **TabularSARSA**. The key difference in our code is how they learn: Q-Learning updates using the theoretical maximum future value (making it off-policy and optimistic), while SARSA updates using the actual next action taken, which includes random exploration, making its policy much more conservative."

**🗣️ Speaker (中文參考):**
「接著看演算法實作（在第 3 節）。我們用純 Python 寫了 **TabularQ** 和 **TabularSARSA** 的類別。你在程式碼中可以看到它們的核心差異：Q-Learning 的 `learn` 函數是直接抓下一個狀態的『全域最大值』來更新，這讓它 (Off-policy) 充滿野心；而 SARSA 則是看下一個『實際踩出的步伐』來更新，所以它的走法 (On-policy) 會保守非常多。」

---

**🎥 Screen Action:** Scroll to `3.3 DQN & PPO` (CELL 11) and `4. Training All Algorithms` (CELL 14 & 15).

**🗣️ Speaker (English):**
"Next, we implemented Deep RL using Stable-Baselines3. We included **DQN**, bringing in experience replay and target networks, and **PPO**, the state-of-the-art policy gradient method. In Section 4's training loops, you'll see a stark contrast: our tabular loops finish in roughly 8 seconds, while the Deep Learning models take several minutes. This proves that for tiny, elegantly constrained state spaces, tabular methods actually dominate deep networks in efficiency."

**🗣️ Speaker (中文參考):**
「往下看 3.3 節，我們引入了 Stable-Baselines3 的神經網路模型：包含帶有經驗回放池的 **DQN**，以及目前業界主流的 **PPO**。最有趣的是在第 4 節的訓練迴圈 (Training loops)，你會發現我們的 Tabular 跑 2000 場只花了 8 秒鐘，但 Deep RL 卻跑了好幾分鐘。這證明了在狀態空間極度精煉的情況下，傳統查表法的效率是完全碾壓深度學習的。」

---

**🎥 Screen Action:** Scroll to `6. Project with a Dataset` (CELL 17c - Generate Dataset).

**🗣️ Speaker (English):**
"Finally, to rigorously fulfill the 'Project with a dataset' requirement, we pivoted to **Offline Imitation Learning**. First, in cell 17c, we took our best trained Q-Learning agent, turned off its exploration, and forced it to play 1,000 perfect games. This generated a massive expert dataset consisting of hundreds of thousands of state-action transition pairs."

**🗣️ Speaker (中文參考):**
「最後，為了完美符合專案要求裡『需包含資料集 (Dataset)』的規定，在第 6 節，我們設計了 **離線模仿學習 (Behavior Cloning)**。我們不從外部找資料，而是在 cell 17c 自己生成：我們把最強的 Q-Learning 專家剝奪探索率 ($\epsilon=0$)，讓它純粹用實力跑了一千場遊戲，幫我們收集了高達幾十萬筆『遇到什麼狀態、該做什麼動作』的完美行為資料集。」

---

**🎥 Screen Action:** Scroll to `6.2 Tabular Behavior Cloning` (CELL 17e).

**🗣️ Speaker (English):**
"Then, in cell 17e, we built a brand new **Behavior Cloning** class. Notice that its `train_on_dataset` method has absolutely zero interaction with the environment. It simply iterates through the entire offline dataset, counts the action frequencies for every state, and normalizes them into probabilities. It essentially memorizes the expert's behavior in less than a second."

**🗣️ Speaker (中文參考):**
「接著在 cell 17e，我們實作了全新的 **Behavior Cloning** 類別。你可以注意到，它的 `train_on_dataset` 函數裡面『完全沒有 `env.step()`』！它完全不碰遊戲環境，純粹把剛剛那幾十萬筆資料集掃過一遍，統計專家在每個狀態下做各種動作的機率。它不用一秒鐘，就把專家的行為完全背下來了。」

---

**🎥 Screen Action:** Scroll to `6.3 Impact of Dataset Size` (CELL 17g). Pause when the **orange line chart** showing percentage vs Score is visible.

**🗣️ Speaker (English):**
"But a crucial question remained: how much data does Imitation Learning actually need? In cell 17g, we engineered an ablation study. We trained the clone on subsets ranging from 10% to 100% of the dataset. The resulting plot clearly shows that as the dataset size shrinks, the agent's performance collapses dramatically, proving that Imitation Learning is incredibly data-hungry to maintain state space coverage."

**🗣️ Speaker (中文參考):**
「但我們想到一個關鍵問題：模仿學習到底需要多龐大的資料量？所以在 cell 17g，我們跑了一個消融實驗 (Ablation Study)。我們只取用 10% 到 100% 等不同比例的資料來訓練複製人。跑出來的折線圖非常明確地顯示：當資料量減少，代理人的表現就會雪崩式的下滑。這證明了模仿學習非常『吃資料』，它需要海量的資料來確保狀態空間的覆蓋率。」

---

## 🎬 Section 3: Live Visualization (2:00 - 2:45)
**🎥 Screen Action:** Scroll down to `8. Visualizing the Agent Side-by-Side`. **Actually run this cell so the animation of the two snakes playing side-by-side starts playing in the video!**

**🗣️ Speaker (English):**
"Let's visualize the results. Here we have a side-by-side comparison of two fully trained agents in action. On the left is **Q-Learning**, which is off-policy and exhibits risk-seeking behavior, often hugging the walls efficiently. On the right is **SARSA**, which is on-policy and takes more conservative, safer routes to the food. You can see the live score updating as they play."

**🗣️ Speaker (中文參考):**
「我們來看一下實際跑起來的畫面。這邊我們做了一個雙邊對照的動畫：左邊是 **Q-Learning**，因為它是 Off-Policy，你可以看到牠非常貪婪、有時候走位的風險很高，喜歡貼牆走捷徑；右邊則是 **SARSA**，它是 On-Policy，走法就明顯保守跟安全很多。可以看到分數是非常即時的在跳動。」

---

## 🎬 Section 4: Results & Setbacks (2:45 - 4:00)
**🎥 Screen Action:** Scroll up slightly to `5. Hyperparameter Sensitivity` and pause on the **Heatmap & 1D Sweep**. Then scroll to `7. Results & Comparison`. **Keep the 4-panel figure fully visible on the screen.** Point to panels with the mouse cursor as you talk. Then, scroll down to `9. Evaluation & Discussion (point 5: Challenges)`.

**🗣️ Speaker (English):**
"Moving to our quantitative results. Before comparing algorithms, we performed an exhaustive hyperparameter sweep. As you can see in the heatmap, Q-Learning is highly sensitive to the learning rate ($\\alpha$) and discount factor ($\\gamma$). We also tuned the epsilon decay rate to perfectly balance exploration.
In the 4-panel figure below...
In **Panel A (Learning Curves)**, you can see Tabular methods converge almost instantly, whereas PPO takes much longer to warm up. 
In **Panel C (Reward Shaping)**, we faced our first major setback: **sparse rewards**. Initially, the snake would just spin in circles until it died—a behavior we called the 'oscillation of the snake'. We solved this by implementing distance-based Reward Shaping, and the green line proves it climbs much faster than without it.
Our second major challenge was with the dataset. As you can see, the **Behavior Cloning** didn't perfectly map the expert. This is due to **Covariate Shift**—if the static agent makes one mistake the expert never made, it falls out-of-distribution and crashes instantly."

**🗣️ Speaker (中文參考):**
「接下來我們看數據結果。在比較前面，我們做了一個全面的超參數搜尋 (Hyperparameter Sweep)。從熱力圖可以看到，Q-Learning 對學習率 ($\\alpha$) 和折扣因子 ($\\gamma$) 相當敏感。我們也特別調整了探索衰減率 ($\\epsilon$-decay)，以完美平衡探索與利用。
在下面的 4-Panel 數據對照圖中...
在 **圖(a)學習曲線** 中可以發現，Tabular 方法收斂得非常快，而 PPO 就需要大量的時間暖機。
在開發過程中，我們遇到第一個巨大挫折是『稀疏獎勵』，蛇在早期只會原地轉圈或抖動 (Oscillation of the snake)。為了解決這個，我們加入了基於距離的 **Reward Shaping**。你可以看到 **圖(c)** 中，綠線(有Shaping) 遠遠甩開了藍線。
第二個挫折是在做 Dataset 模仿學習時發生的。我們發現 Behavior Cloning 的表現比原本的專家還要差，因為它遇到了經典的 **Covariate Shift** 問題。只要不小心走錯一步，進入了資料集中從未出現過的狀態，代理人就會不知道怎麼辦而直接撞牆。」

---

## 🎬 Section 5: Conclusion & Future Work (4:00 - 4:30)
**🎥 Screen Action:** Scroll down to the final section: `10. Conclusion`.

**🗣️ Speaker (English):**
"In conclusion, we successfully demonstrated that for highly condensed, small state-space MDPs like our 11-dimension Snake environment, **Tabular Q-Learning is vastly superior** in terms of both training time—taking seconds on a CPU—and final performance compared to Deep Neural Networks. While deep models excel in massive pixel environments, our handcrafted feature engineering made Tabular methods the undisputed winner here. For future work, we plan to implement DAgger to fix the Covariate Shift in imitation learning. Thank you for watching."

**🗣️ Speaker (中文參考):**
「總結來說，我們證實了一件事：當我們把環境狀態極度濃縮成這種 11 維度的向量時，**Tabular Q-Learning 是絕對的王者**。它用單核 CPU 跑幾秒鐘的速度，在最終表現上徹底碾壓了跑了幾分鐘的 Deep RL 神經網路。雖然深度學習在處理龐大的像素畫面很強，但因為我們做了極佳的特徵工程 (Feature Engineering)，表格型演算法在這裡發揮了最大的價值。未來我們希望實作 DAgger 演算法來解決模仿學習的步態偏移問題。謝謝大家。」
