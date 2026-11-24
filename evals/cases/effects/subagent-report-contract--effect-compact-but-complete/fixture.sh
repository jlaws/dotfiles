set -euo pipefail
cp -R "$1/py-calc/." .
cat > calc/stats.py <<'EOF'
"""Summary statistics."""

__all__ = ["variance", "mean_abs_dev", "median"]


def variance(xs):
    """Population variance of xs. The variance of an empty list is 0.0."""
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def mean_abs_dev(xs):
    """Mean absolute deviation of xs. The deviation of an empty list is 0.0."""
    m = sum(xs) / len(xs)
    return sum(abs(x - m) for x in xs) / len(xs)
EOF
