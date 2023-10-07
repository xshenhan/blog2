---
redirect_from: _posts/2023-10-06-Denoising Diffusion Implicit Models
title: Denoising Diffusion Implicit Models
tags:
    - Diffusion
    - Apex Lab 
---
# Denoising Diffusion Implicit Models

## **Background**

- Denoising Diffusion Probabilistic Models

## **Variational inference for non-Markovian forward processes**

### **non-Markovian forward process**

1. Let us consider a family Q of inference distributions, indexed by a real vector $\sigma\in R^T_{\ge0}$:
    
    $$
    \begin{equation}q_{\sigma}\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_{0}\right):=q_{\sigma}\left(\boldsymbol{x}_{T} \mid \boldsymbol{x}_{0}\right) \prod_{t=2}^{T} q_{\sigma}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, \boldsymbol{x}_{0}\right)\end{equation}
    $$
    
    where $q_\sigma(\boldsymbol x_T|\boldsymbol x_0) = \mathcal N (\sqrt \alpha_T \boldsymbol x_0, (1-\alpha_T)\boldsymbol I)$ and for all $t\gt 1$
    
    $$
    \begin{equation}q_{\sigma}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, \boldsymbol{x}_{0}\right)=\mathcal{N}\left(\sqrt{\alpha_{t-1}} \boldsymbol{x}_{0}+\sqrt{1-\alpha_{t-1}-\sigma_{t}^{2}} \cdot \frac{\boldsymbol{x}_{t}-\sqrt{\alpha_{t}} \boldsymbol{x}_{0}}{\sqrt{1-\alpha_{t}}}, \sigma_{t}^{2} \boldsymbol{I}\right)\end{equation}
    $$
    
    **Then**, we can proof that $q_\sigma (\boldsymbol x_t|\boldsymbol x_0) = \mathcal N(\sqrt \alpha_t\boldsymbol x_0,(1-\alpha_t)\boldsymbol I)$ holds for all t. (The mean function in (2) is chosen to order to ensure this). The forward process can be derived from Baye's rule:
    
    $$
    \begin{equation}q_{\sigma}\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_{0}\right)=\frac{q_{\sigma}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, \boldsymbol{x}_{0}\right) q_{\sigma}\left(\boldsymbol{x}_{t} \mid \boldsymbol{x}_{0}\right)}{q_{\sigma}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{0}\right)}\end{equation}
    $$
    
    前向过程仍然是高斯分布，但是不具有马尔科夫性。
    

## **Generative process and unified variational inference objective**

下面，我们来定义一个可训练的生成过程 $p_\theta(x_{0:T})$，其中每一步的 $p_\theta^{(t)}(\boldsymbol x_{t-1}|\boldsymbol x_t)$利用到了$q_\sigma(\boldsymbol x_{t-1}|\boldsymbol x_t, \boldsymbol x_0)$中的信息

对于$\boldsymbol x_0 \sim q(\boldsymbol x_0) 和 \epsilon_t\sim \mathcal N(\boldsymbol 0, \boldsymbol I)$, 我们可以计算得到$\boldsymbol x_t$。在不告诉模型$\boldsymbol x_0$的情况下，利用模型$\epsilon_\theta^{(t)}(\boldsymbol x_t)$预测第t步$\boldsymbol x_t$上被加上的噪声$\epsilon_t$，我们进而就可以计算得到$\boldsymbol x_0$的一个估计值：

$$
\begin{equation}f_{\theta}^{(t)}\left(x_{t}\right):=\left(x_{t}-\sqrt{1-\alpha_{t}} \cdot \epsilon_{\theta}^{(t)}\left(x_{t}\right)\right) / \sqrt{\alpha_{t}}.\end{equation}
$$

于是，我们便可以用一个固定的点 $p_\theta(\boldsymbol x_T)= \mathcal N(\boldsymbol 0,\boldsymbol I)$ 定义一个生成过程:

$$
\begin{equation}p_{\theta}^{(t)}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}\right)=\left\{\begin{array}{ll}\mathcal{N}\left(f_{\theta}^{(1)}\left(\boldsymbol{x}_{1}\right), \sigma_{1}^{2} \boldsymbol{I}\right) & \text { if } t=1 \\ q_{\sigma}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, f_{\theta}^{(t)}\left(\boldsymbol{x}_{t}\right)\right) & \text { otherwise }\end{array}\right.\end{equation}
$$

其中 $q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, f_{\theta}^{(t)}\left(\boldsymbol{x}_{t}\right)\right)$ 是(2)中定义的

**目标函数**：

$$
\begin{equation}\begin{aligned} & J_{\sigma}\left(\epsilon_{\theta}\right):=\mathbb{E}_{\boldsymbol{x}_{0: T} \sim q_{\sigma}\left(\boldsymbol{x}_{0: T}\right)}\left[\log q_{\sigma}\left(\boldsymbol{x}_{1: T} \mid \boldsymbol{x}_{0}\right)-\log p_{\theta}\left(\boldsymbol{x}_{0: T}\right)\right] \\ = & \mathbb{E}_{\boldsymbol{x}_{0: T} \sim q_{\sigma}\left(\boldsymbol{x}_{0: T}\right)}\left[\log q_{\sigma}\left(\boldsymbol{x}_{T} \mid \boldsymbol{x}_{0}\right)+\sum_{t=2}^{T} \log q_{\sigma}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{t}, \boldsymbol{x}_{0}\right)-\sum_{t=1}^{T} \log p_{\theta}^{(t)}\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_{\ell}\right)-\log p_{\theta}\left(\boldsymbol{x}_{T}\right)\right]\end{aligned}\end{equation}
$$