# CartPolePolicyGradient

Policy Gradient Search
P(left | s) = 1 / (1 + exp(-theta^T x))
P(right | s) = 1 - 1 / (1 + exp(-theta^T x))

We want to find the gradient with respect to theta of
E[f(tau)].

This is equal to the expected value of the gradient with respect to theta of log P_theta(tau) multipled with f(tau)
