from collections import defaultdict


class MetricSink:
    """Simple in-memory metric aggregator for scaffold runs."""

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def log(self, name: str, value: float) -> None:
        self._values[name].append(float(value))

    def mean(self, name: str) -> float:
        values = self._values.get(name, [])
        if not values:
            return 0.0
        return sum(values) / len(values)

    def summary(self) -> dict[str, float]:
        return {key: self.mean(key) for key in sorted(self._values.keys())}
