import importlib
def load_strategy(strategy_name):
    module = importlib.import_module(f"strategies.{strategy_name}")
    print(f"Loaded strategy module: strategies.{strategy_name}")
    return module.Strategy()