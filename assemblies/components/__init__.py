"""Proven components: parameterised parts with their geometry, their models and their CFD/FEA case builders.

A component fills an ``assemblies.vida.Assembly`` node (``record`` results, ``attach`` files); it knows nothing about
the workflows that use it. Components import the Vegeta tools and other components only, never ``notebooks/``.
"""
