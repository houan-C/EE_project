# 影像重建實驗量化評估指標：數學原理與運算解析指南

本文件詳細闡述在無線視訊傳輸與深度學習影像重建實驗（`Realtime_Simulator.py`）中所採用的四大核心指標：**邊緣銳利度 (Sharpness)**、**結構相似性 (SSIM)**、**知覺感知損失 (LPIPS)** 以及 **每秒幀率 (FPS)** 之數學推導、運算步驟與物理意涵。

---

## 一、 邊緣銳利度 (Sharpness) —— 拉普拉斯變異數法 (Laplacian Variance)

### 1. 數學本質
在數位影像處理中，物體的邊緣、輪廓與高頻紋理代表**像素強度在空間上的劇烈跳變（Gradient 變化）**。為衡量影像邊緣的鮮銳程度，最常用的二階空間微分算子為**拉普拉斯算子 (Laplacian Operator, $\nabla^2$)**：

$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

在離散影像像素座標中，拉普拉斯算子透過一個 $3 \times 3$ 的卷積核（Kernel）實作：

$$K = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

* **平滑/模糊區域**：相鄰像素差值極小，二階導數近乎為零。
* **高對比/銳利邊緣**：像素梯度劇烈跳變，二階導數產生顯著的正負波峰與波谷。

### 2. 計算公式
程式將灰階影像與拉普拉斯卷積核做二維卷積後，計算全圖二階響應值的**統計變異數 (Variance)**：

$$\text{Sharpness} = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum_{x, y} \Big( (\nabla^2 I)(x, y) - \mu_{\nabla^2 I} \Big)^2$$

其中：
* $N$ 為影像像素總數。
* $\mu_{\nabla^2 I}$ 為拉普拉斯卷積後的像素均值。

### 3. 物理意義與數值解讀
* **未經 AI 優化 (Baseline, 典型值 20 ~ 50)**：因 TX 端壓縮下採樣，再經 RX 傳統雙線性插值放大，高頻信號被平均抹除，邊緣變平緩發虛，變異數極低。
* **RealESRGAN 重建後 (典型值 300 ~ 800+)**：超解析度生成網路重建了清晰的線條與輪廓，二階導函數的能量大幅飆升，銳利度提升數倍至十數倍。

---

## 二、 結構相似性 (SSIM, Structural Similarity Index)

### 1. 核心設計理念
傳統的均方誤差 (MSE) 與峰值信噪比 (PSNR) 僅單純計算逐像素的絕對數值差異，無法反映人類視覺系統 (HVS) 對場景結構的敏感度。SSIM（由 Zhou Wang 等人於 2004 年提出）將影像比對解耦為三個獨立的感知維度：

1. **亮度對比 (Luminance, $l$)**：評估局部平均灰階強度。
2. **對比度 (Contrast, $c$)**：評估局部標準差（動態範圍大小）。
3. **結構度 (Structure, $s$)**：評估去除均值與標準差後之幾何形狀相關性。

### 2. 局部計算公式
在大小為 $N \times N$（通常為 $11 \times 11$ 高斯加權窗口）的對應局部視窗 $x$ 與 $y$ 內：

$$\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma$$

通常令 $\alpha = \beta = \gamma = 1$，化簡為：

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

其中：
* $\mu_x, \mu_y$：局部視窗 $x$ 與 $y$ 的像素平均值。
* $\sigma_x^2, \sigma_y^2$：局部視窗 $x$ 與 $y$ 的方差。
* $\sigma_{xy}$：視窗 $x$ 與 $y$ 的互協方差 (Cross-covariance)。
* $C_1 = (K_1 L)^2, C_2 = (K_2 L)^2$：穩定常數（$L=255, K_1=0.01, K_2=0.03$），防止分母接近零時產生數值不穩定。

### 3. 全局平均 (Mean SSIM)
$$\text{MSSIM}(X, Y) = \frac{1}{M} \sum_{j=1}^M \text{SSIM}(x_j, y_j)$$

### 4. 學理解析：感知－失真權衡 (Perception-Distortion Tradeoff)
在 GAN 超解析度任務中，常出現「肉眼看極度清晰，但 SSIM 數值持平或微降 0.1% ~ 0.5%」的現象：
* **數學定論**：Blau & Michaeli 在 CVPR 2018 證明的理論指出，**感知品質（Perception）與失真度（Distortion）存在不可兼得的理論邊界**。
* **原因**：生成對抗網路（GAN）會合成極度逼真自然的高頻紋理（如金屬反光、草地細紋），但這些合成像素無法做到 100% 絕對座標與原圖噪聲重合。像素級比對（SSIM）對輕微位置位移極為敏感，故數值不增加，但視覺真實度（LPIPS / Sharpness）大幅提升。

---

## 三、 知覺感知失真度 (LPIPS, Learned Perceptual Image Patch Similarity)

### 1. 核心設計理念
LPIPS 由 UC Berkeley 的 Richard Zhang 等人於 CVPR 2018 發表。該指標利用在百萬張影像上預訓練的深度卷積神經網路（如 AlexNet），將影像映射至高維特徵空間，模擬人腦視覺皮層各層級神經元的激發反應。

### 2. 計算流程架構
```
[輸入影像 1 (原圖 x)]   ──> [AlexNet 卷積層] ──> 特徵圖抽取 ──> 通道單位化 (Unit-norm) ┐
                                                                                 ├──> 乘上權重向量 w ──> 計算加權 L2 距離
[輸入影像 2 (重建圖 x₀)] ──> [AlexNet 卷積層] ──> 特徵圖抽取 ──> 通道單位化 (Unit-norm) ┘
```

### 3. 數學計算公式
$$d(x, x_0) = \sum_{l=1}^L \frac{1}{H_l W_l} \sum_{h, w} \left\| w_l \odot \left( \hat{y}_{hw}^l - \hat{y}_{0, hw}^l \right) \right\|_2^2$$

其中：
* $l$：AlexNet 中的特徵抽取層（通常取 conv1 至 conv5 共 5 層）。
* $H_l, W_l$：第 $l$ 層特徵圖的空間高度與寬度。
* $\hat{y}^l, \hat{y}_0^l$：經 Channel 維度單位化（Unit-length normalized）後的特徵張量。
* $w_l$：經由大規模人類雙盲偏好實驗（BAPPS 資料集，包含超過 48.4 萬對人類主觀判定）學習得到的層級縮放權重向量。
* $\odot$：Hadamard 逐元素乘積。

### 4. 物理意義與數值解讀
* **特性**：$0 \sim 1$ 之間，**數值越小越好（0 代表人類大腦完全感知不出差異）**。
* **實驗成效**：從未經 AI 的 0.54 顯著降低至 0.47（降低 13.2%），證明神經網路高維特徵距離被拉近，消除人眼厭惡的模糊感與馬賽克瑕疵。

---

## 四、 畫面幀率 (FPS) 與 RIFE 光流補幀運算

### 1. 傳輸端頻寬限制模型
在無線圖傳鏈路中，由於通道調變與傳輸延遲，每秒發送的有效訊框受限。若預設頻寬上限為 $\text{FPS}_{\text{TX}}$，且封包成功率為 $\text{PSR}$，則接收端實際取得的基準幀率為：

$$\text{FPS}_{\text{Received}} = \text{FPS}_{\text{TX}} \times \text{PSR}$$

例如在 $0.20 \text{ km}$ 下：$16.45 \times 99.5\% \approx 16.37 \text{ FPS}$。

### 2. RIFE 深度光流插幀機制
接收端快取相鄰兩幀有效影像 $I_t$ 與 $I_{t+1}$，輸入至 RIFE 神經網路：

1. **多尺度雙向光流預測 (IFNet)**：預測由 $t \to t+1$ 與 $t+1 \to t$ 的密集運動向量場：
   $$F_{t \to t+1}, \quad F_{t+1 \to t}$$
2. **逆向扭曲與特徵融合 (Backward Warping & Context Fusion)**：
   $$I_{t+0.5} = \mathcal{M} \odot \text{Warp}(I_t, F_{t \to t+0.5}) + (1 - \mathcal{M}) \odot \text{Warp}(I_{t+1}, F_{t+1 \to t+0.5}) + \Delta$$
   其中 $\mathcal{M}$ 為遮擋融合權重圖，$\Delta$ 為高頻殘差補償項。
3. **影格序列翻倍輸出**：
   $$I_t \longrightarrow I_{t+0.5} \longrightarrow I_{t+1}$$
   $$\text{FPS}_{\text{Reconstructed}} = \text{FPS}_{\text{Received}} \times 2 = 16.37 \times 2 = 32.74 \text{ FPS}$$

---

## 五、 四大指標協同分析速查總結表

| 指標名稱 | 英文全稱 | 數學算子 / 演算法基礎 | 評估面向 | 理想方向 | 實驗中 AI 效益體現 |
|:---|:---|:---|:---|:---:|:---|
| **Sharpness** | Laplacian Variance | 二階空間微分卷積核之變異數 | 局部高頻邊緣強度 | **越高越好** | **暴增 16.57 倍**，證明 RealESRGAN 生成了豐富且銳利的高反差邊緣 |
| **LPIPS** | Learned Perceptual Image Patch Similarity | 預訓練 AlexNet 深度感知特徵加權 $L_2$ 距離 | 人眼主觀感知失真度 | **越低越好** | **改善 13.2%**，證明超解析度細節自然逼真，消除壓縮模糊 |
| **SSIM** | Structural Similarity Index | 局部滑動視窗之均值、標準差與互相關分析 | 全局幾何結構與輪廓保真度 | **越近 1 越好** | **維持約 0.70 (微降 0.3%)**，印證感知－失真權衡理論，結構未畸變 |
| **FPS** | Frames Per Second | 時間軸影格輸出速率統計 | 視訊動態流暢度 | **越高越好** | **由 16.37 翻倍至 32.74 FPS (+100%)**，突破物理頻寬限制，達到舒適流暢門檻 |
