# Triton XPU metadata alias

Torch XPU installs the importable `triton` module from the distribution named
`triton-xpu`. Some pure-Python packages require a distribution named `triton`.
This empty wheel satisfies that metadata requirement and depends on the exact
`triton-xpu` version. It contains no Python modules or native code.
