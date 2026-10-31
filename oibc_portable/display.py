"""Formatting is separate from metric precision and ranking."""
from decimal import Decimal, ROUND_HALF_UP, localcontext
import math


def won(value):
    if value is None: return "—"
    if isinstance(value,(list,tuple)): return "["+", ".join(won(v) for v in value)+"]"
    if not math.isfinite(float(value)): return "—"
    number=Decimal(str(value))
    with localcontext() as context:
        context.prec=max(28,max(0,number.adjusted())+len(number.as_tuple().digits)+3)
        rounded = number.quantize(Decimal("1"),rounding=ROUND_HALF_UP)
    return format(int(rounded),",")
