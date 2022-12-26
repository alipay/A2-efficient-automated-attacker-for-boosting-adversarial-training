from auto_adv.operations import *

PRIMITIVES = [
    FGM,
    FGSM,
    FGMM,
    FGSMM,
    FGMAdvM,
    Identity,
    Gaussian,
    Uniform,
]

PRIMITIVES_VAL = [
    'FGM',
    'FGSM',
    'FGMM',
    'FGSMM',
    'FGMAdvM',
    'Identity',
    'Gaussian',
    'Uniform',
]

GRAD_PRIMITIVES = [
    FGM,
    FGSM,
    FGMM,
    FGSMM,
    FGMAdvM,
    Identity
]

GRAD_PRIMITIVES_VAL = [
    'FGM',
    'FGSM',
    'FGMM',
    'FGSMM',
    'FGMAdvM',
    'Identity'
]

RAND_PRIMITIVES = [
    Gaussian,
    Uniform,
    Identity
]

RAND_PRIMITIVES_VAL = [
    'Gaussian',
    'Uniform',
    'Identity'
]


LRS = [
    0.001,
    0.01,
    0.1,
    1.,
    # 10.,
]

LRS_VAL = [
    0.001,
    0.01,
    0.1,
    1.,
    # 10.,
]
