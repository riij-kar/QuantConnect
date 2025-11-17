def build_indicators(algo, symbol, config):
    """Build and return indicator instances.

    QuantConnect (pythonnet) imports are done inside this function so the module can be
    imported in a regular Python environment (without CLR/pythonnet) for linting or tests.
    """
    # local import to avoid importing QuantConnect at module import time
    from QuantConnect.Indicators import (
        MovingAverageType,
        ExponentialMovingAverage,
        RelativeStrengthIndex,
        BollingerBands,
        AverageTrueRange,
        VolumeWeightedAveragePriceIndicator,
    )

    periods = config["indicator_params"]
    indicators = {}
    for period in periods["ema_periods"]:
        indicators[f"ema_{period}"] = ExponentialMovingAverage(period)
    indicators["rsi"] = RelativeStrengthIndex(periods["rsi_period"])
    indicators["bb"] = BollingerBands(periods["bb_period"], periods["bb_std_dev"])
    indicators["atr"] = AverageTrueRange(periods["atr_period"], MovingAverageType.Simple)
    indicators["vwap"] = VolumeWeightedAveragePriceIndicator(periods["vwap_period"])
    algo.Debug(f"Built indicators1: {list(indicators)}")
    algo.Debug(f"Built indicators2: {indicators}")
    return indicators