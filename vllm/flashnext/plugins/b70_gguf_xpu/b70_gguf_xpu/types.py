"""ggml type ids used by the Flash-Next IQ3_XXS GGUF mix."""

Q2_K = 10
Q3_K = 11
Q4_K = 12
Q5_K = 13
Q6_K = 14
IQ2_XXS = 16
IQ2_XS = 17
IQ3_XXS = 18
IQ4_NL = 20
IQ3_S = 21
IQ2_S = 22
IQ4_XS = 23
Q2_0 = 42

# (block elements, block bytes). Row bytes = (K / block_elems) * block_bytes.
BLOCK = {
    Q4_K: (256, 144),
    Q5_K: (256, 176),
    Q6_K: (256, 210),
    IQ2_XXS: (256, 66),
    IQ2_XS: (256, 74),
    IQ3_XXS: (256, 98),
    IQ4_NL: (32, 18),
    IQ3_S: (256, 110),
    IQ2_S: (256, 82),
    IQ4_XS: (256, 136),
    Q2_K: (256, 84),
    Q3_K: (256, 110),
    Q2_0: (64, 18),
}

NAMES = {
    Q4_K: "Q4_K",
    Q5_K: "Q5_K",
    Q6_K: "Q6_K",
    IQ2_XXS: "IQ2_XXS",
    IQ2_XS: "IQ2_XS",
    IQ3_XXS: "IQ3_XXS",
    IQ4_NL: "IQ4_NL",
    IQ3_S: "IQ3_S",
    IQ2_S: "IQ2_S",
    IQ4_XS: "IQ4_XS",
    Q2_K: "Q2_K",
    Q3_K: "Q3_K",
    Q2_0: "Q2_0",
}


def row_bytes(ggml_type: int, k: int) -> int:
    be, bb = BLOCK[ggml_type]
    if k <= 0 or k % be != 0:
        return 0
    return (k // be) * bb
