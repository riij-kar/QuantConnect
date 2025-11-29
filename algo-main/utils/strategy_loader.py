import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Optional
@dataclass
class StrategyBundle:
    """Container holding strategy and optional framework components.

    Attributes
    ----------
    strategy : Any
        Primary strategy instance that exposes framework lifecycle hooks.
    alpha_model : Any, optional
        Alpha model supplied by the strategy module, already instantiated when
        possible.
    execution_model : Any, optional
        Execution model configured for the algorithm framework.
    portfolio_model : Any, optional
        Portfolio construction model that controls position sizing.
    risk_model : Any, optional
        Risk management component implementing protective rules.
    """

    strategy: Any
    alpha_model: Optional[Any] = None
    execution_model: Optional[Any] = None
    portfolio_model: Optional[Any] = None
    risk_model: Optional[Any] = None


def _try_instantiate(attr: Any) -> Optional[Any]:
    """Return an instantiated object when possible, otherwise the attribute itself.

    Parameters
    ----------
    attr : Any
        Member retrieved from a strategy module. It can be a class, callable,
        already-instantiated object, or ``None``.

    Returns
    -------
    Any or None
        Instance of ``attr`` when instantiation succeeds, the original object
        when instantiation is not required or fails due to missing parameters,
        or ``None`` when ``attr`` was ``None``.
    """

    if attr is None:
        return None
    if inspect.isclass(attr):
        try:
            return attr()
        except TypeError:
            return attr
    if callable(attr):
        try:
            return attr()
        except TypeError:
            # Attribute might already be an instance or require constructor args.
            return attr
    return attr


def load_strategy(strategy_name: str) -> StrategyBundle:
    """Import a strategy module and materialize its framework components.

    Parameters
    ----------
    strategy_name : str
        Name of the module inside the ``strategies`` package.

    Returns
    -------
    StrategyBundle
        Data class bundling the strategy instance together with optional
        alpha, execution, portfolio, and risk models.

    Raises
    ------
    AttributeError
        If the target module does not expose a ``Strategy`` attribute.
    """
    module = importlib.import_module(f"strategies.{strategy_name}")
    print(f"Loaded strategy module: strategies.{strategy_name}")

    strategy = _try_instantiate(getattr(module, "Strategy", None))
    if strategy is None:
        raise AttributeError(
            f"Strategy module 'strategies.{strategy_name}' must define a Strategy class or instance"
        )

    return StrategyBundle(
        strategy=strategy,
        alpha_model=_try_instantiate(getattr(module, "AlphaModel", None)),
        execution_model=_try_instantiate(getattr(module, "ExecutionModel", None)),
        portfolio_model=_try_instantiate(getattr(module, "PortfolioModel", None)),
        risk_model=_try_instantiate(getattr(module, "RiskModel", None)),
    )