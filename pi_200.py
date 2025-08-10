from decimal import Decimal, getcontext
import math


def compute_pi_chudnovsky(num_decimal_places: int) -> Decimal:
    """Compute pi using the Chudnovsky algorithm to the requested decimal places.

    num_decimal_places: Number of digits after the decimal point.
    Returns a Decimal approximation of pi with sufficient precision.
    """
    # Add guard digits to keep precision during intermediate calculations
    guard_digits = 10
    getcontext().prec = num_decimal_places + guard_digits

    # Chudnovsky constants
    c = Decimal(426880) * Decimal(10005).sqrt()

    # Number of terms required (~14.18 digits per term); add a small safety margin
    terms = max(1, (num_decimal_places // 14) + 2)

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
    """Return pi formatted to exactly num_decimal_places after the decimal point."""
    pi = compute_pi_chudnovsky(num_decimal_places)
    # Format with fixed number of decimal places
    return f"{pi:.{num_decimal_places}f}"


if __name__ == "__main__":
    print(format_pi(200))