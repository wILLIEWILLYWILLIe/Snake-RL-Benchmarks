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
"But a crucial question remained: how much data does Imitation Learning actually need? In cell 17g, we engineered an ablation study. We trained the clone on subsets ranging from 10% to 100% of the dataset. The resulting plot clearly shows that **performance plateaus very early**. Even with just 10% of the data—about 40,000 samples—the simple tabular agent already achieves an average score of around 24, which is nearly as good as the full dataset. This suggests that for our compact 11-dimensional state space, a small volume of expert data provides nearly exhaustive coverage."

**🗣️ Speaker (中文參考):**
「但我們想到一個關鍵問題：模仿學習到底需要多龐大的資料量？所以在 cell 17g，我們跑了一個消融實驗 (Ablation Study)。我們只取用 10% 到 100% 等不同比例的資料來訓練複製人。跑出來的折線圖非常明確地顯示：**表現很快就進入了高原期 (Plateau)**。即使只用 10% 的資料——大約四萬筆樣本——這個簡單的查表法代理人平均分數就已經來到 24 分左右，跟用全量資料差不多。這證明了在我們的 11 維精簡狀態空間下，少量的資料就足以涵蓋大部分的情況。」

---

## 🎬 Section 3: Live Visualization (2:00 - 2:45)
**🎥 Screen Action:** Scroll down to `8. Visualizing the Agent Side-by-Side`. **Actually run this cell so the animation of the two snakes playing side-by-side starts playing in the video!**

**🗣️ Speaker (English):**
"Let's visualize the results. Here we have a side-by-side comparison of two fully trained agents in action. On the left is **Q-Learning**, which is off-policy and exhibits risk-seeking behavior, often hugging the walls efficiently. On the right is our **SARSA (or Deep RL)** agent. While they both play effectively, the Q-Learning agent is slightly more aggressive, while the on-policy nature of SARSA makes it a bit more cautious. You can see the live score updating as they play."

**🗣️ Speaker (中文參考):**
「我們來看一下實際跑起來的畫面。這邊我們做了一個雙邊對照的動畫：左邊是 **Q-Learning**，因為它是 Off-Policy，你可以看到牠非常貪婪、喜歡貼牆走捷徑；右邊則是 **SARSA (或是 Deep RL)**，它的走法就明顯保守跟安全一些。可以看到分數是非常即時的在跳動。」

---

## 🎬 Section 4: Results & Setbacks (2:45 - 4:00)
**🎥 Screen Action:** Scroll up slightly to `5. Hyperparameter Sensitivity` and pause on the **Heatmap & 1D Sweep**. Then scroll to `7. Results & Comparison`. **Keep the 4-panel figure fully visible on the screen.** Point to panels with the mouse cursor as you talk. Then, scroll down to `9. Evaluation & Discussion (point 5: Challenges)`.

**🗣️ Speaker (English):**
"Moving to our quantitative results. Before comparing algorithms, we performed an exhaustive hyperparameter sweep. As you can see in the heatmap, Q-Learning is highly sensitive to its parameters. Our primary finding is that **high $\gamma$ (0.95 or above)** is absolutely critical; a lower discount factor makes the snake short-sighted, leading to the low-scoring 'red zones' you see in the corner. We also tuned the epsilon decay rate to perfectly balance exploration.
In the 4-panel figure below...
In **Panel (a)**, the learning curves show Tabular methods converge almost instantly, while PPO takes more time to reach peak performance.
In **Panel (b)**, looking at final average scores, you'll notice that **Q-Learning, SARSA, and PPO** all achieve mastery with scores above 20, whereas **DQN** noticeably lags behind in this environment.
In **Panel (c)**, we show the impact of **Reward Shaping**. Without it, the snake often aimlessly moves in circles—a behavior we call the 'oscillation of the snake'. Adding distance-based rewards solves this, allowing the agent to climb much faster.
Finally, **Panel (d)** shows the score distribution. You'll see that while Q-Learning has a high average, its variance is a bit wider than SARSA's, which tends to be more consistent due to its conservative nature."

**🗣️ Speaker (中文參考):**
「接下來我們看數據結果。在比較前，我們做了一個全面的超參數搜尋 (Hyperparameter Sweep)。從熱力圖可以看到，Q-Learning 對參數非常敏感。我們最重要的發現是：**高 $\gamma$ (0.95 以上)** 是絕對關鍵的；較低的折扣因子會讓蛇變得很短視，導致你在角落看到的低分『紅區』。我們也特別調整了探索衰減率 ($\epsilon$-decay)，以平衡探索與利用。
讓我們看這張四合一的數據圖表：
在 **圖(a)** 中，你可以看到 Tabular 方法幾乎是瞬間收斂，而 PPO 則需要更長的時間才達到最高水準。
在 **圖(b)** 的最終平均分中，**Q-Learning、SARSA 和 PPO** 都穩定達到 20 分以上，表現優異；相比之下，**DQN** 在這個環境中的效率明顯較差。
在 **圖(c)** 中，我們展示了 **Reward Shaping** 的威力。沒有它，蛇只會在那裡原地打轉（我們稱之為「蛇的震盪」）。加入距離獎勵後，學習曲線明顯拉升。
最後，**圖(d)** 顯示了分數分佈。你可以看到 Q-Learning 雖然高分，但波動稍微比 SARSA 大一點，SARSA 因為走法保守，表現相對穩定。」

---

## 🎬 Section 5: Conclusion & Future Work (4:00 - 4:30)
**🎥 Screen Action:** Scroll down to the final section: `10. Conclusion`.

**🗣️ Speaker (English):**
"In conclusion, we successfully implemented and compared four Online RL algorithms and one Offline Imitation Learning algorithm. Our key takeaways are:
First, **Tabular methods excel** when the state space is elegantly condensed. Q-Learning achieved the highest performance in mere seconds, outperforming Deep RL models that took minutes.
Second, we observed the fundamental difference between **Off-policy and On-policy** learners: Q-Learning's optimistic updates lead to aggressive, risk-taking behavior, while SARSA remains more cautious.
Third, while **Imitation Learning** is extremely fast to train, it remains brittle due to **Covariate Shift** when it faces states outside its training data.
And finally, the project proved that **Reward Shaping and Hyperparameter tuning** are not just secondary details—they are critical to getting RL agents to converge at all.
For future work, we plan to implement DAgger to fix imitation shift and expand our model to **Raw Pixel learning** using CNNs to better utilize the power of Deep RL. Thank you for watching."

**🗣️ Speaker (中文參考):**
「總結來說，我們成功實作並比較了四種在線強化學習與一種離線模仿學習。我們有幾個關鍵的收穫：
第一，當環境狀態被精煉得很好時，**Tabular 方法表現優異**。Q-Learning 只需要幾秒鐘就能達到最高效能，遠快於需要數分鐘的深度學習模型。
第二，我們觀察到了 **Off-policy 與 On-policy** 的本質差異：Q-Learning 的樂觀更新讓牠展現出富有侵略性的冒險走法，而 SARSA 則相對謹慎穩定。
第三，**模仿學習 (BC)** 雖然訓練極快，但因為 **Covariate Shift** 問題，在面對資料集以外的狀態時顯得非常脆弱。
最後，這個專案證明了 **Reward Shaping 和 超參數調優** 並非可有可無，而是讓強化學習收斂的關鍵工程手段。
未來，我們計畫實作 DAgger 演算法來優化模仿學習，並嘗試使用 CNN 進行 **Raw Pixel (原始圖像)** 的學習，以更全面地發揮深度學習的優勢。謝謝大家。」
