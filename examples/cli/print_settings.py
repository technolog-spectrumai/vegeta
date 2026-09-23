"""Mellonia settings for the demo (generic PLA example with a finer layer height)."""
from vegeta.mellonia.examples import GENERIC_PLA_0_2MM

FINE = GENERIC_PLA_0_2MM.replace(print={"layer_height": 0.15, "first_layer_height": 0.2})
