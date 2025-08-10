from decimal import Decimal, getcontext, ROUND_DOWN
import math


def compute_pi_chudnovsky(num_decimal_places: int) -> Decimal:
    """Compute pi using the Chudnovsky algorithm to the requested decimal places.

    num_decimal_places: Number of digits after the decimal point.
    Returns a Decimal approximation of pi with sufficient precision.
    """
    # Add guard digits to keep precision during intermediate calculations
    # Use a larger guard to ensure correct rounding at the requested precision
    guard_digits = 20
    getcontext().prec = num_decimal_places + guard_digits

    # Chudnovsky constants
    c = Decimal(426880) * Decimal(10005).sqrt()

    # Number of terms required (~14.18 digits per term); add a safety margin
    # Use ceil to ensure enough terms
    terms = max(1, math.ceil(num_decimal_places / 14) + 2)

    series_sum = Decimal(0)

    # Use integers for factorial computations; convert to Decimal at division time
    for k in range(terms):
        six_k_fact = math.factorial(6 * k)
        three_k_fact = math.factorial(3 * k)
        k_fact = math.factorial(k)

        numerator = six_k_fact * (13591409 + 545140134 * k)
        denominator = three_k_fact * (k_fact ** 3) * ((-262537412640768000) ** k)

        term = Decimal(numerator) / Decimal(denominator)
        series_sum += term

    pi = c / series_sum
    return pi


def format_pi(num_decimal_places: int) -> str:
    """Return pi formatted to exactly num_decimal_places after the decimal point.

    Uses truncation (ROUND_DOWN) to avoid rounding up on the last digit.
    """
    # Compute with extra precision so quantize can truncate accurately
    pi = compute_pi_chudnovsky(num_decimal_places + 2)
    quant = Decimal(1).scaleb(-num_decimal_places)  # 10^{-num_decimal_places}
    pi_trunc = pi.quantize(quant, rounding=ROUND_DOWN)
    # Format with fixed number of decimal places
    return f"{pi_trunc:.{num_decimal_places}f}"


if __name__ == "__main__":
    print(format_pi(200))