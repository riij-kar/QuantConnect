def build_indicators(algo, symbol, config):
    """Build and return indicator instances.

    Parameters
    ----------
    algo : QCAlgorithm
        Algorithm instance that will own the indicators. Used here for
        diagnostic logging.
    symbol : Symbol
        Security symbol whose data will feed the indicators.
    config : dict
        Configuration section expected to expose an ``indicator_params``
        mapping that lists periods for each indicator family.

    Returns
    -------
    dict
        Dictionary keyed by indicator name (for example ``"ema_9"``) pointing
        to newly constructed QuantConnect indicator objects.

    Notes
    -----
    QuantConnect (pythonnet) imports are performed inside the function so the
    module can still be imported during local linting or unit testing without a
    CLR runtime.
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