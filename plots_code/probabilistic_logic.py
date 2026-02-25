from abc import ABC

import torch

LOG_EPS = 1e-6
STABLE_PRODUCT_EPS = 1e-2
log_logistic = torch.nn.LogSigmoid()


class Product():
    """
    Inspired from Badreddine, et al. (2020). Logic Tensor Networks, https://arxiv.org/abs/2012.13635

    For the conjunction and negation
    """

    DEFAULT_GRAD = 50.0

    @staticmethod
    def greater(
        x1: torch.Tensor, x2: torch.Tensor | float, grad: float = DEFAULT_GRAD
    ) -> torch.Tensor:
        return log_logistic(grad * (x1 - x2))  # type: ignore

    @staticmethod
    def smaller(
        x1: torch.Tensor, x2: torch.Tensor | float, grad: float = DEFAULT_GRAD
    ) -> torch.Tensor:
        return log_logistic(grad * (x2 - x1))  # type: ignore

    @classmethod
    def and_(cls, *x: torch.Tensor) -> torch.Tensor:
        """
        If multiple tensors are provided, the element wise 'and' is returned
        If a single tensor is given, then the logical 'and' reduces all dimensions except the first
        """
        if len(x) == 1:
            return x[0].flatten(1).sum(1)
        return torch.stack(x).sum(0)

    @classmethod
    def or_(cls, *x: torch.Tensor, eps=LOG_EPS) -> torch.Tensor:
        if len(x) == 1:
            return x[0]
        if len(x) == 2:
            small = torch.min(x[0], x[1])
            big = torch.max(x[0], x[1])
            return torch.log(1 + torch.exp(small - big) - torch.exp(small)) + big

        return cls.or_(cls.or_(x[0], x[1]), *x[2:])
    
    @classmethod
    def lse_or_(cls, *x: torch.Tensor, eps=LOG_EPS) -> torch.Tensor:
        """
        Numerically stable log-sum-exp version of the or operation
        """
        return torch.logsumexp(torch.stack(x), dim=0)   

    @classmethod
    def not_(cls, x: torch.Tensor) -> torch.Tensor:
        """
        This doesn't work numerically, very hard problem to solve
        """
        return torch.log(1 - torch.exp(x) + LOG_EPS)

    @classmethod
    def in_(
        cls,
        x: torch.Tensor,
        inf: torch.Tensor | float,
        sup: torch.Tensor | float,
        grad: float = DEFAULT_GRAD,
    ) -> torch.Tensor:
        return cls.and_(cls.greater(x, inf, grad), cls.smaller(x, sup, grad))  # type: ignore

    @classmethod
    def equal(
        cls, x1: torch.Tensor, x2: torch.Tensor | float, grad: float = DEFAULT_GRAD
    ) -> torch.Tensor:
        return cls.in_(x1, inf=x2, sup=x2, grad=grad)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import math
    
    x = torch.linspace(-0.1, 0.1, steps=1000)
    x_exp = torch.linspace(-1, 1, steps=1000)
    grads = [5.0, 10.0, 25.0, 50, 100]

    cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=min(math.log(g) for g in grads), vmax=max(math.log(g) for g in grads))

    # Plotting the greater function

    plt.subplot(2, 2, 1)
    for grad in grads:
        color = cmap(norm(math.log(grad)))
        plt.plot(x.numpy(), Product.greater(x, 0, grad=grad).detach().numpy(), label=f"grad={grad}", color=color, alpha=0.8)
    plt.title("$c[x>0]$")
    plt.grid(linestyle="--", alpha=0.7)

    plt.subplot(2, 2, 2)
    for grad in grads:
        color = cmap(norm(math.log(grad)))
        plt.plot(x_exp.numpy(), Product.greater(x_exp, 0, grad=grad).exp().detach().numpy(), label=f"grad={grad}", color=color, alpha=0.8)
    plt.title("$e^{c[x>0]}$")
    plt.grid(linestyle="--", alpha=0.7)

    # Plotting the equal function
    plt.subplot(2, 2, 3)
    for grad in grads:
        color = cmap(norm(math.log(grad)))
        plt.plot(x.numpy(), Product.equal(x, 0, grad=grad).detach().numpy(), label=f"grad={grad}", color=color, alpha=0.8)
    plt.title("$c[x=0]$")
    plt.grid(linestyle="--", alpha=0.7)

    plt.subplot(2, 2, 4)
    for grad in grads:
        color = cmap(norm(math.log(grad)))
        plt.plot(x_exp.numpy(), Product.equal(x_exp, 0, grad=grad).exp().detach().numpy(), label=f"grad={grad}", color=color, alpha=0.8)
    plt.title("$e^{c[x=0]}$")
    plt.grid(linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)

    fig = plt.gcf()
    handles = [
        mpl.lines.Line2D([0], [0], color=cmap(norm(math.log(g))), lw=2, alpha=0.8)
        for g in grads
    ]
    labels = [f"k={int(g)}" if float(g).is_integer() else f"k={g}" for g in grads]
    fig.legend(handles, labels, loc="lower center", ncol=len(grads), frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.savefig("images/log_probabilistic_functions.pdf", bbox_inches="tight", dpi=600)
    
    plt.close()
    
    
    
    # Plot the formula functions
    def formula_or(x, k=50.0):
        return Product.or_(
            Product.greater(x, 0.5, grad=k),
            Product.smaller(x, -0.5, grad=k),
        )
        
    def formula_and(x, k=50.0):
        return Product.and_(
            Product.smaller(x, 0.5, grad=k),
            Product.greater(x, -0.5, grad=k),
        )
        
    def alternative_or(x, k=50.0):
        return Product.lse_or_(
            Product.greater(x, 0.5, grad=k),
            Product.smaller(x, -0.5, grad=k),
        )
    
    x = torch.linspace(-2, 2, steps=1000)
    grads = [5.0, 10.0, 25.0, 50, 100]
    
    cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=min(math.log(g) for g in grads), vmax=max(math.log(g) for g in grads))
    
    # Plotting the formula functions
    
    # copy figsize from previous plot
    plt.figure(figsize=(6, 5))
    

    
    plt.subplot(2, 2, 1)
    for grad in grads:
        color = cmap(norm(math.log(grad)))
        plt.plot(x.numpy(), formula_or(x, k=grad).detach().numpy(), label=f"k={grad}", color=color, alpha=0.8)
    plt.title(r"$c[x < -0.5 \vee x > 0.5]$")
    plt.grid(linestyle="--", alpha=0.7)
    
    plt.subplot(2, 2, 2)
    for grad in grads:
        color = cmap(norm(math.log(grad)))
        plt.plot(x.numpy(), formula_or(x, k=grad).exp().detach().numpy(), label=f"k={grad}", color=color, alpha=0.8)
    plt.title(r"$e^{c[x < -0.5 \vee x > 0.5]}$")
    plt.grid(linestyle="--", alpha=0.7)
    
    plt.subplot(2, 2, 3)
    for grad in grads:
        color = cmap(norm(math.log(grad)))
        plt.plot(x.numpy(), formula_and(x, k=grad).detach().numpy(), label=f"k={grad}", color=color, alpha=0.8)
    plt.title(r"$c[x > -0.5 \wedge x < 0.5]$")
    plt.grid(linestyle="--", alpha=0.7)
    
    plt.subplot(2, 2, 4)
    for grad in grads:
        color = cmap(norm(math.log(grad)))
        plt.plot(x.numpy(), formula_and(x, k=grad).exp().detach().numpy(), label=f"k={grad}", color=color, alpha=0.8)
    plt.title(r"$e^{c[x > -0.5 \wedge x < 0.5]}$")
    plt.grid(linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)
    
    fig = plt.gcf()
    handles = [
        mpl.lines.Line2D([0], [0], color=cmap(norm(math.log(g))), lw=2, alpha=0.8)
        for g in grads
    ]
    labels = [f"k={int(g)}" if float(g).is_integer() else f"k={g}" for g in grads]
    fig.legend(handles, labels, loc="lower center", ncol=len(grads), frameon=False, bbox_to_anchor=(0.5, -0.02))
    
    plt.savefig("images/log_probabilistic_formulas.pdf", bbox_inches="tight", dpi=600)
    plt.close()

