import os
from ast import literal_eval

from dotenv import load_dotenv
from tensorflow import keras

load_dotenv()

N_FACTORS = int(os.getenv("N_FACTORS"))
ITEM_MODEL_SHAPE = literal_eval(os.getenv("ITEM_MODEL_SHAPE"))
USER_META_MODEL_SHAPE = literal_eval(os.getenv("USER_META_MODEL_SHAPE"))
USER_INTERACTION_MODEL_SHAPE = literal_eval(os.getenv("USER_INTERACTION_MODEL_SHAPE"))


def item_model():
    inp = keras.layers.Input(shape=ITEM_MODEL_SHAPE)

    layer_1 = keras.layers.Dense(
        N_FACTORS,
        activation="elu",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        activity_regularizer=keras.regularizers.l2(l2=1e-6),
    )(inp)

    layer_2 = keras.layers.Dense(
        N_FACTORS,
        activation="elu",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        activity_regularizer=keras.regularizers.l2(l2=1e-6),
    )(layer_1)

    add = keras.layers.Add()([layer_1, layer_2])

    out = keras.layers.Dense(
        N_FACTORS,
        activation="linear",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        activity_regularizer=keras.regularizers.l2(l2=1e-6),
    )(add)

    return keras.models.Model(inp, out)


def user_model():
    inp_meta = keras.layers.Input(shape=USER_META_MODEL_SHAPE)
    inp_interaction = keras.layers.Input(shape=USER_INTERACTION_MODEL_SHAPE)

    layer_1_meta = keras.layers.Dense(
        N_FACTORS,
        activation="elu",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        activity_regularizer=keras.regularizers.l2(l2=1e-6),
    )(inp_meta)

    layer_1_interaction = keras.layers.Dense(
        N_FACTORS,
        activation="elu",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        activity_regularizer=keras.regularizers.l2(l2=1e-6),
    )(inp_interaction)

    layer_2_meta = keras.layers.Dense(
        N_FACTORS,
        activation="elu",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        activity_regularizer=keras.regularizers.l2(l2=1e-6),
    )(layer_1_meta)

    add = keras.layers.Add()([layer_1_meta, layer_2_meta])

    concat_meta_interaction = keras.layers.Concatenate()([add, layer_1_interaction])

    out = keras.layers.Dense(
        N_FACTORS,
        activation="linear",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        activity_regularizer=keras.regularizers.l2(l2=1e-6),
    )(concat_meta_interaction)

    return keras.models.Model([inp_meta, inp_interaction], out)
