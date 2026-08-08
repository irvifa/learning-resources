# Unit 5, Lesson 1: Gradients and Multivariate Optimisation

---

## Overview

This lesson reinterprets gradients through the lens of optimisation. Rather than viewing gradients purely as measures of sensitivity, we study them as **geometric objects** that determine descent directions and characterise stationary points in multivariate objective functions.

**Learning goal:** Interpret gradients geometrically and explain how they determine the structure of multivariate optimisation problems.

---

## 1. Motivation: Optimisation as Structured Descent

Learning in AI is framed as minimising an objective function over model parameters:

$$\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)$$

where $\theta \in \mathbb{R}^n$ represents model parameters and $\mathcal{L} : \mathbb{R}^n \to \mathbb{R}$ is a loss or objective function.

Optimisation transforms calculus into a **decision rule**: at any point $\theta$, which direction should we move to decrease $\mathcal{L}$ most effectively?

---

## 2. The Gradient as a Geometric Object

### 2.1 Gradient Definition

For $\mathcal{L} : \mathbb{R}^n \to \mathbb{R}$, the gradient at a point $\theta$ is:

$$\nabla \mathcal{L}(\theta) = \begin{pmatrix} \dfrac{\partial \mathcal{L}}{\partial \theta_1} \\ \vdots \\ \dfrac{\partial \mathcal{L}}{\partial \theta_n} \end{pmatrix} \in \mathbb{R}^n$$

### 2.2 Directional Derivatives

The directional derivative of $\mathcal{L}$ at $\theta$ in direction $d \in \mathbb{R}^n$ (with $\|d\| = 1$) is:

$$D_d \mathcal{L}(\theta) = \lim_{h \to 0} \frac{\mathcal{L}(\theta + hd) - \mathcal{L}(\theta)}{h} = \nabla \mathcal{L}(\theta)^\top d$$

This is the **rate of change** of $\mathcal{L}$ in direction $d$.

### 2.3 Steepest Descent Direction

By the Cauchy–Schwarz inequality:

$$\nabla \mathcal{L}(\theta)^\top d \geq -\|\nabla \mathcal{L}(\theta)\| \cdot \|d\|$$

with equality when $d = -\dfrac{\nabla \mathcal{L}(\theta)}{\|\nabla \mathcal{L}(\theta)\|}$. Therefore, the **steepest descent direction** is:

$$d^* = -\nabla \mathcal{L}(\theta)$$

---

## 3. Geometry of Level Sets and Contours

The **level set** (or contour) of $\mathcal{L}$ at value $c$ is:

$$\mathcal{C}_c = \{\theta \in \mathbb{R}^n : \mathcal{L}(\theta) = c\}$$

**Key geometric fact:** The gradient $\nabla \mathcal{L}(\theta)$ is **orthogonal** to the level set $\mathcal{C}_{\mathcal{L}(\theta)}$ at $\theta$, and points in the direction of **steepest ascent**.

For a direction $v$ tangent to the level set:

$$\nabla \mathcal{L}(\theta)^\top v = 0$$

---

## 4. First-Order Optimality Conditions

### Necessary Condition for a Local Minimum

If $\theta^*$ is a local minimiser of $\mathcal{L}$ and $\mathcal{L}$ is differentiable at $\theta^*$, then:

$$\nabla \mathcal{L}(\theta^*) = \mathbf{0}$$

A point satisfying this is called a **stationary point**. Note: the converse does not hold in general — stationary points may be saddle points or local maxima.

### Descent Direction Condition

A direction $d$ is a **descent direction** at $\theta$ if:

$$\nabla \mathcal{L}(\theta)^\top d < 0$$

This guarantees that there exists $\alpha > 0$ small enough such that:

$$\mathcal{L}(\theta + \alpha d) < \mathcal{L}(\theta)$$

---

## 5. Dual Role of Gradients in Optimisation

| Role | Description |
|------|-------------|
| **Local approximation** | First-order Taylor expansion: $\mathcal{L}(\theta + \delta) \approx \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top \delta$ |
| **Movement rule** | Algorithm design: move in direction $-\nabla \mathcal{L}(\theta)$ to decrease $\mathcal{L}$ |

The **first-order Taylor approximation** at $\theta$:

$$\mathcal{L}(\theta + \delta) \approx \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top \delta + O(\|\delta\|^2)$$

This local linear model underpins gradient-based algorithm design.

---

## 6. Connection to Gradient Descent

The basic gradient descent update rule follows directly:

$$\theta_{k+1} = \theta_k - \alpha_k \nabla \mathcal{L}(\theta_k)$$

where $\alpha_k > 0$ is the **step size** (learning rate). This is a special case of the general line search framework:

$$\theta_{k+1} = \theta_k + \alpha_k d_k, \quad d_k = -\nabla \mathcal{L}(\theta_k)$$

---

## Core Reading

| Text | Sections |
|------|----------|
| Nocedal & Wright, *Numerical Optimization* | Ch. 2, §2.1–2.2 — solutions, local minima, necessary conditions, steepest descent |
| Boyd & Vandenberghe, *Convex Optimization* | Ch. 1, §1.1–1.2 — problem formulation, learning as structured minimisation |

---

## Lesson Outline

- **1.1** Review of gradients in multivariate settings
- **1.2** Directional derivatives and descent directions
- **1.3** Geometry of level sets and contours
- **1.4** First-order optimality conditions
- **1.5** Summary and forward links

---

## 1.1 Review of Gradients in Multivariate Settings

### Local Linear Approximation

For a differentiable function $\mathcal{L} : \mathbb{R}^n \to \mathbb{R}$, the gradient provides a local linear model near any point $\theta$:

$$\mathcal{L}(\theta + \delta) \approx \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top \delta$$

where $\delta \in \mathbb{R}^n$ is a small perturbation and $\nabla \mathcal{L}(\theta)^\top \delta$ is the first-order change in $\mathcal{L}$. Near any point, the objective behaves approximately like a tilted plane whose slope is given by the gradient. This linearisation is the mathematical foundation of gradient-based algorithms; they take small steps within the region where the linear approximation is informative.

### Magnitude of the Gradient

The norm $\|\nabla \mathcal{L}(\theta)\|$ measures the maximum instantaneous rate of increase of the function. A large gradient indicates a steep region where the function changes rapidly; a small gradient indicates a flat region where changes are slow. Flat regions can slow optimisation dramatically even when far from the optimum — a point returned to in Lesson 3 when analysing convergence behaviour.

### Gradient as Direction of Maximum Increase

For a unit direction $d$ with $\|d\| = 1$, the directional derivative measures the rate of change of $\mathcal{L}$ in direction $d$:

$$D_d \mathcal{L}(\theta) = \nabla \mathcal{L}(\theta)^\top d$$

By the Cauchy–Schwarz inequality, $\nabla \mathcal{L}(\theta)^\top d \leq \|\nabla \mathcal{L}(\theta)\|$, with equality when $d = \dfrac{\nabla \mathcal{L}(\theta)}{\|\nabla \mathcal{L}(\theta)\|}$. The gradient therefore points in the direction of maximum increase, and the direction of steepest descent is:

$$d^* = -\frac{\nabla \mathcal{L}(\theta)}{\|\nabla \mathcal{L}(\theta)\|}$$

The contour lines represent level sets of the objective — all points sharing the same function value. The gradient at any point is perpendicular to the contour and points toward steepest increase. The negative gradient points toward steepest descent, which is why gradient-based methods move opposite to the gradient.

### Example 1: Simple Quadratic Bowl

Let $\mathcal{L}(\theta) = \theta_1^2 + \theta_2^2$. Then:

$$\nabla \mathcal{L}(\theta) = \begin{pmatrix} 2\theta_1 \\ 2\theta_2 \end{pmatrix}$$

At the point $\theta = (1, 1)^\top$, the gradient is $\nabla \mathcal{L} = (2, 2)^\top$ and the steepest descent direction is $d^* = -(1/\sqrt{2}, 1/\sqrt{2})^\top$. The function increases radially outward from the origin, so descent always moves toward the origin regardless of starting point — a property that makes this a well-behaved optimisation landscape.

### Example 2: Non-Quadratic Objective

Let $\mathcal{L}(\theta) = \theta_1^2 + \sin(\theta_2)$. Then:

$$\frac{\partial \mathcal{L}}{\partial \theta_1} = 2\theta_1, \qquad \frac{\partial \mathcal{L}}{\partial \theta_2} = \cos(\theta_2)$$

Here the gradient direction varies nonlinearly with $\theta$; the slope in the $\theta_2$-direction oscillates as $\cos(\theta_2)$, meaning the descent direction changes shape across the domain. This illustrates a key difference from quadratic objectives: in non-quadratic problems, the gradient must be recomputed at each step because the local linear approximation only holds near the current point.

---

## 1.2 Directional Derivatives and Descent Directions

### Directional Derivative

The directional derivative of $\mathcal{L}$ along a unit vector $d$ (with $\|d\| = 1$) is:

$$D_d \mathcal{L}(\theta) = \nabla \mathcal{L}(\theta)^\top d$$

It measures the instantaneous rate of change of $\mathcal{L}$ in the direction $d$. When this quantity is positive, the function increases in that direction; when negative, it decreases; when zero, there is no first-order change.

### Descent Directions

A vector $d$ is a **descent direction** if $\nabla \mathcal{L}(\theta)^\top d < 0$, meaning the angle between $d$ and $\nabla \mathcal{L}(\theta)$ exceeds $90°$ and moving in direction $d$ decreases the function. The negative gradient $d = -\nabla \mathcal{L}(\theta)$ is always a descent direction whenever $\nabla \mathcal{L}(\theta) \neq \mathbf{0}$, since:

$$\nabla \mathcal{L}(\theta)^\top (-\nabla \mathcal{L}(\theta)) = -\|\nabla \mathcal{L}(\theta)\|^2 < 0$$

### Example 3: Weighted Quadratic

Let $\mathcal{L}(\theta) = \theta_1^2 + 4\theta_2^2$. Then:

$$\frac{\partial \mathcal{L}}{\partial \theta_1} = 2\theta_1, \qquad \frac{\partial \mathcal{L}}{\partial \theta_2} = 8\theta_2$$

$$\nabla \mathcal{L}(\theta) = \begin{pmatrix} 2\theta_1 \\ 8\theta_2 \end{pmatrix}$$

The gradient component in the $\theta_2$-direction is four times larger than in the $\theta_1$-direction for the same displacement. The function penalises $\theta_2$ more strongly, so a descent step adjusts $\theta_2$ more aggressively than $\theta_1$. This **anisotropy** is visible in the elliptical contours, which reflect stronger curvature along $\theta_2$ than $\theta_1$.

The elliptical contours are stretched along the $\theta_1$-axis because the function changes more slowly in that direction. The gradient at any point is perpendicular to the local contour, while the tangent direction lies along it — confirming that moving along a level set produces no first-order change in the objective.

This geometric imbalance between directions has direct consequences for optimisation: when curvature differs strongly across axes, gradient descent steps that work well in one direction may be ineffective or unstable in another. Understanding this anisotropy is essential for interpreting convergence behaviour, developed formally in Lesson 3.

---

## 1.3 Geometry of Level Sets and Contours

### Level Sets

A level set is the set of all points where the objective takes a constant value:

$$\mathcal{C}_c = \{\theta \in \mathbb{R}^n : \mathcal{L}(\theta) = c\}$$

In two dimensions, these appear as contour curves; in higher dimensions, they are surfaces. Moving along a level set does not change the function value; the level set is precisely the set of directions along which the first-order change is zero.

### Gradient Orthogonality

If $v$ is tangent to a level set at $\theta$, then $\nabla \mathcal{L}(\theta)^\top v = 0$; the gradient is **perpendicular** to the level set. This follows directly from the definition: moving along the level set leaves $\mathcal{L}$ unchanged to first order, so the directional derivative in any tangent direction must be zero:

$$\nabla \mathcal{L}(\theta)^\top v = 0 \quad \text{for all } v \text{ tangent to } \mathcal{C}_{\mathcal{L}(\theta)} \text{ at } \theta$$

The gradient therefore points in the only direction orthogonal to all tangent directions — the direction of steepest change. This geometric relationship explains why gradient descent crosses contours orthogonally rather than following them.

### Stationary Points

A stationary point satisfies $\nabla \mathcal{L}(\theta^*) = \mathbf{0}$. At such a point, no first-order descent direction exists, and the linear approximation of $\mathcal{L}$ is flat. Stationary points may be local minima, local maxima, or saddle points, so satisfying $\nabla \mathcal{L}(\theta^*) = \mathbf{0}$ is a **necessary but not sufficient** condition for optimality. Distinguishing between these cases requires second-order information — the Hessian developed in Unit 3 Lesson 3 — which returns in Lesson 4 when we study convexity.

---

## 1.4 First-Order Optimality Conditions

For an unconstrained differentiable optimisation problem $\min_\theta \mathcal{L}(\theta)$, a necessary condition for local optimality is that the gradient vanishes at the solution:

$$\nabla \mathcal{L}(\theta^*) = \mathbf{0}$$

This is the **first-order optimality condition**. Its intuition is immediate: if $\nabla \mathcal{L}(\theta^*) \neq \mathbf{0}$, then moving in the direction $-\nabla \mathcal{L}(\theta^*)$ would decrease the objective, so $\theta^*$ cannot be a local minimum. A local minimum must therefore be a point where no descent direction exists, and since the negative gradient is always a descent direction when the gradient is nonzero, the gradient must be zero.

### Necessary but Not Sufficient

The first-order condition is **necessary but not sufficient** for optimality. It must hold at every local minimum, but it may also hold at saddle points and local maxima. Satisfying $\nabla \mathcal{L}(\theta^*) = \mathbf{0}$ identifies a stationary point — a candidate for optimality — but does not confirm what kind of stationary point it is.

### Example 4: A Saddle Point Satisfies the First-Order Condition

Consider $f(\theta_1, \theta_2) = \theta_1^2 - \theta_2^2$. The gradient is:

$$\frac{\partial f}{\partial \theta_1} = 2\theta_1, \qquad \frac{\partial f}{\partial \theta_2} = -2\theta_2$$

$$\nabla f(\theta) = \begin{pmatrix} 2\theta_1 \\ -2\theta_2 \end{pmatrix}$$

At the origin, $\nabla f(\mathbf{0}) = \mathbf{0}$, so the first-order condition is satisfied. Yet the origin is **not** a minimum: moving in the $\theta_1$-direction increases $f$, while moving in the $\theta_2$-direction decreases it. The function has a saddle shape — it curves upward in one direction and downward in another.

### Second-Order Conditions

To classify a stationary point, we examine the Hessian $H = \nabla^2 \mathcal{L}(\theta^*)$:

| Hessian condition | Classification |
|---|---|
| $H \succ 0$ (positive definite): $v^\top H v > 0$ for all $v \neq \mathbf{0}$ | Local minimum |
| $H \prec 0$ (negative definite): $v^\top H v < 0$ for all $v \neq \mathbf{0}$ | Local maximum |
| $H$ has mixed-sign eigenvalues | Saddle point |

In the bowl example $f = \theta_1^2 + \theta_2^2$, the Hessian is $H = 2I$, which is positive definite with eigenvalues $\lambda_1 = \lambda_2 = 2$. In the saddle example $f = \theta_1^2 - \theta_2^2$, the Hessian is:

$$H = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$$

with eigenvalues $\lambda_1 = 2 > 0$ and $\lambda_2 = -2 < 0$ — mixed signs confirming a saddle point.

In **convex optimisation** (Lesson 4), this classification simplifies dramatically: any stationary point of a convex function is automatically a **global minimum**. This is one of the most powerful consequences of convexity and one of the main reasons convex problems are so much easier to solve reliably than non-convex ones.

---

## 1.5 Summary and Forward Links

In this lesson, gradients were reinterpreted through the lens of optimisation:

- The **local linear approximation** $\mathcal{L}(\theta + \delta) \approx \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top \delta$ shows that the gradient is the coefficient vector determining how the objective changes in any direction.
- The **norm** $\|\nabla \mathcal{L}(\theta)\|$ measures the maximum instantaneous rate of increase; large gradients indicate steep regions, small gradients indicate flat ones.
- The **directional derivative** $D_d \mathcal{L}(\theta) = \nabla \mathcal{L}(\theta)^\top d$ extends this to arbitrary unit directions. By Cauchy–Schwarz, the maximum rate of increase occurs exactly in the gradient direction, making $-\nabla \mathcal{L}(\theta)$ the direction of steepest descent. A vector $d$ is a descent direction whenever $\nabla \mathcal{L}(\theta)^\top d < 0$.
- **Level sets** satisfy $\nabla \mathcal{L}(\theta)^\top v = 0$ for tangent directions $v$, confirming that the gradient is perpendicular to the level set at every point. This orthogonality explains why gradient descent crosses contours rather than following them, and why elliptical contours cause gradient descent to behave differently in different directions.
- At a **stationary point** where $\nabla \mathcal{L}(\theta^*) = \mathbf{0}$, the first-order optimality condition is satisfied, but optimality is not guaranteed. Stationary points may be local minima, local maxima, or saddle points — distinguishing them requires the Hessian.

### Forward Links

| Lesson | Connection |
|--------|-----------|
| **Lesson 2** | Loss functions and risk minimisation as the concrete objectives that optimisation acts on |
| **Lesson 3** | Gradient descent as an iterative algorithm; convergence analysis in terms of gradient geometry |
| **Lesson 4** | Convexity: when stationary points are guaranteed global minima, transforming local geometry into global guarantees |
