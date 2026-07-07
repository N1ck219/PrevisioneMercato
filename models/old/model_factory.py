import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Attention,
                                     concatenate, MultiHeadAttention,
                                     BatchNormalization, LayerNormalization, Flatten)
from tensorflow.keras.optimizers import Adam
from core.models.informer_layers import PositionalEncoding, ProbSparseAttention, DistillationLayer, SeriesDecomposition, MultiScaleDistillation

def build_v4_3_model(input_shape):
    """Architettura per la versione 4.3"""
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True, name="lstm_base")(inputs)
    x = Dropout(0.2)(x)
    att = Attention(name="att")([x, x])
    x = LSTM(64, name="lstm_feat")(att)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
    return model

def build_v4_6_model(input_shape):
    """Architettura profonda per le versioni 4.6 e Crypto 1.7"""
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True, name="lstm_base")(inputs)
    x = Dropout(0.3)(x)
    att = Attention(name="att")([x, x])
    x = LSTM(64, name="lstm_feat")(att)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy')
    return model

def build_v5_0_split_brain_model(shape_t, shape_m):
    """Architettura a doppio input (Split Brain) per V5.6 e V6.4"""
    in_t = Input(shape=shape_t)
    x_t = LSTM(128, return_sequences=True)(in_t)
    x_t = Dropout(0.3)(x_t)
    att = Attention()([x_t, x_t])
    x_t = LSTM(64)(att)
    
    in_m = Input(shape=shape_m)
    x_m = LSTM(64)(in_m) 
    
    merged = concatenate([x_t, x_m])
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[in_t, in_m], outputs=out)
    return model

def build_v7_0_split_brain_model(shape_t, shape_m):
    """Architettura Split Brain V7.0 ottimizzata per GPU e trading intraday.
    
    Evoluzione del concetto Split Brain con:
    - LSTM più profondi (256→128 ramo tecnico) per catturare pattern intraday
    - Multi-Head Attention (2 heads) per analisi multi-scala temporale
    - BatchNormalization per stabilità del training
    - Configurazione CuDNN-compatible per massima performance GPU
    
    Ramo Tecnico (VWAP + indicatori + sessione):
        Input → LSTM(256, return_seq) → Dropout → MultiHeadAttention(2h)
              → LayerNorm → LSTM(128) → Dropout
    
    Ramo Macro (rendimenti indici + flag sessione):
        Input → LSTM(64) → Dropout
    
    Merger:
        Concatenate → Dense(128) → BatchNorm → Dropout
                    → Dense(64) → Dropout → Dense(1, sigmoid)
    """
    # ── Ramo Tecnico (VWAP, RSI, Bollinger, ATR, session) ──
    in_t = Input(shape=shape_t, name='tech_input')
    # LSTM CuDNN-compatible: no recurrent_dropout, activation='tanh', recurrent_activation='sigmoid'
    x_t = LSTM(256, return_sequences=True, name='tech_lstm_1')(in_t)
    x_t = Dropout(0.3, name='tech_drop_1')(x_t)
    
    # Multi-Head Attention per catturare pattern a diverse scale intraday
    att_t = MultiHeadAttention(num_heads=2, key_dim=128, name='tech_mha')(x_t, x_t)
    x_t = LayerNormalization(name='tech_layer_norm')(att_t + x_t)  # Residual connection
    
    x_t = LSTM(128, name='tech_lstm_2')(x_t)
    x_t = Dropout(0.2, name='tech_drop_2')(x_t)
    
    # ── Ramo Macro (NASDAQ ret, VIX ret, etc. + flag sessione) ──
    in_m = Input(shape=shape_m, name='macro_input')
    x_m = LSTM(64, name='macro_lstm')(in_m)
    x_m = Dropout(0.2, name='macro_drop')(x_m)
    
    # ── Merge e Decision Layer ──
    merged = concatenate([x_t, x_m], name='brain_merge')
    x = Dense(128, activation='relu', name='decision_1')(merged)
    x = BatchNormalization(name='decision_bn')(x)
    x = Dropout(0.3, name='decision_drop_1')(x)
    x = Dense(64, activation='relu', name='decision_2')(x)
    x = Dropout(0.2, name='decision_drop_2')(x)
    out = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=[in_t, in_m], outputs=out, name='SplitBrain_V7_0')
    return model

def build_v8_0_informer_model(shape_t, shape_m, lookback=60, pred_horizon=12):
    """Architettura Informer V8.0
    
    Riceve input storici, applica Positional Encoding, ProbSparse Attention e Distillation.
    Emette una singola probabilità di confidenza / previsione direzionale (Double head loss).
    """
    # ── Ramo Tecnico ──
    in_t = Input(shape=shape_t, name='tech_input')
    
    # Projection base
    x_t = Dense(128, name='tech_proj')(in_t)
    x_t = PositionalEncoding(max_steps=lookback, max_dims=128)(x_t)
    
    # Block 1
    att_1 = ProbSparseAttention(num_heads=4, key_dim=32)(x_t, x_t)
    x_t = LayerNormalization()(x_t + Dropout(0.2)(att_1))
    x_t = DistillationLayer(filters=128)(x_t)  # L/2
    
    # Block 2
    att_2 = ProbSparseAttention(num_heads=4, key_dim=32)(x_t, x_t)
    x_t = LayerNormalization()(x_t + Dropout(0.2)(att_2))
    x_t = DistillationLayer(filters=128)(x_t)  # L/4
    
    x_t_flat = Flatten()(x_t)

    # ── Ramo Macro ──
    in_m = Input(shape=shape_m, name='macro_input')
    x_m = LSTM(64, name='macro_lstm')(in_m)
    x_m = Dropout(0.2, name='macro_drop')(x_m)
    
    # ── Merge ──
    merged = concatenate([x_t_flat, x_m], name='brain_merge')
    x = Dense(128, activation='gelu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Double Head Output: 
    # 1. return_pred (Regression) per penalizzare gli errori di ampiezza. 
    # 2. confidence (Sigmoid) per l'uso nel calcolo finale.
    # Useremo return_pred come out target principale per il training (DirectionalLoss), 
    # ed è possibile usare output sigmoidale come 'confidenza' isolato.
    # Ma per far contenti keras.Model.fit su singola truth (y_true), generiamo una sola output
    # con attivazione Tanh per [-1, 1], da cui si estrarrà la confidence mappandolo a [0, 1].
    
    out = Dense(1, activation='tanh', name='return_prediction')(x)
    
    model = Model(inputs=[in_t, in_m], outputs=out, name='Informer_V8_0')
    return model

def build_v8_1_informer_model(shape_t, shape_m, lookback=60):
    """Architettura Informer V8.1 (Ultra-Veloce)
    
    Usa MultiHeadAttention nativa Keras (ottimizzata in C++) invece della custom ProbSparseAttention.
    Perfettamente compatibile con Mixed Precision senza colli di bottiglia CPU/Python.
    """
    # ── Ramo Tecnico ──
    in_t = Input(shape=shape_t, name='tech_input')
    
    # Projection base
    x_t = Dense(128, name='tech_proj')(in_t)
    x_t = PositionalEncoding(max_steps=lookback, max_dims=128)(x_t)
    
    # Block 1 (Self-Attention nativa)
    att_1 = MultiHeadAttention(num_heads=4, key_dim=32, name='mha_1')(x_t, x_t)
    x_t = LayerNormalization()(x_t + Dropout(0.2)(att_1))
    x_t = DistillationLayer(filters=128)(x_t)  # L/2
    
    # Block 2 (Self-Attention nativa)
    att_2 = MultiHeadAttention(num_heads=4, key_dim=32, name='mha_2')(x_t, x_t)
    x_t = LayerNormalization()(x_t + Dropout(0.2)(att_2))
    x_t = DistillationLayer(filters=128)(x_t)  # L/4
    
    x_t_flat = Flatten()(x_t)

    # ── Ramo Macro ──
    in_m = Input(shape=shape_m, name='macro_input')
    x_m = LSTM(64, name='macro_lstm')(in_m)
    x_m = Dropout(0.2, name='macro_drop')(x_m)
    
    # ── Merge ──
    merged = concatenate([x_t_flat, x_m], name='brain_merge')
    x = Dense(128, activation='gelu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    out = Dense(1, activation='tanh', name='return_prediction')(x)
    
    model = Model(inputs=[in_t, in_m], outputs=out, name='Informer_V8_1')
    return model

def build_v8_2_informer_model(shape_t, shape_m, lookback=60):
    """Architettura Informer V8.2 (Decomposition & Multi-Scale)
    
    Aggiunge SeriesDecomposition per analizzare trend e residui separatamente,
    ed usa MultiScaleDistillation per catturare varie frequenze temporali.
    """
    # ── Ramo Tecnico ──
    in_t = Input(shape=shape_t, name='tech_input')
    
    # 1. Decomposizione della Serie
    res_t, trend_t = SeriesDecomposition(kernel_size=25)(in_t)
    
    # Projection base sui residui (più volatili, richiedono l'attention)
    x_t = Dense(128, name='tech_proj')(res_t)
    x_t = PositionalEncoding(max_steps=lookback, max_dims=128)(x_t)
    
    # Block 1: Attention sui Residui + MultiScale Distillation
    att_1 = MultiHeadAttention(num_heads=4, key_dim=32, name='mha_res_1')(x_t, x_t)
    x_t = LayerNormalization()(x_t + Dropout(0.2)(att_1))
    x_t = MultiScaleDistillation(filters=128)(x_t)  # L/2
    
    # Block 2
    att_2 = MultiHeadAttention(num_heads=4, key_dim=32, name='mha_res_2')(x_t, x_t)
    x_t = LayerNormalization()(x_t + Dropout(0.2)(att_2))
    x_t = MultiScaleDistillation(filters=128)(x_t)  # L/4
    
    x_t_flat = Flatten()(x_t)

    # Elaborazione del Trend (meno complesso, basta un pooling o dense)
    trend_flat = Flatten()(trend_t)
    trend_feat = Dense(64, activation='gelu')(trend_flat)

    # ── Ramo Macro ──
    in_m = Input(shape=shape_m, name='macro_input')
    x_m = LSTM(64, name='macro_lstm')(in_m)
    x_m = Dropout(0.2, name='macro_drop')(x_m)
    
    # ── Merge ──
    merged = concatenate([x_t_flat, trend_feat, x_m], name='brain_merge')
    x = Dense(256, activation='gelu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='gelu')(x)
    x = Dropout(0.2)(x)
    
    out = Dense(1, activation='tanh', name='return_prediction')(x)
    
    model = Model(inputs=[in_t, in_m], outputs=out, name='Informer_V8_2')
    return model

def build_v8_3_informer_model(shape_t, shape_m, lookback=60):
    """Architettura Informer V8.3 (Multi-Task Learning)
    
    Introduce una Dual Head Architecture:
    1. direction_output (Sigmoid) per la classificazione direzionale (su/giù)
    2. magnitude_output (Tanh) per la regressione dell'ampiezza del movimento
    """
    # ── Ramo Tecnico ──
    in_t = Input(shape=shape_t, name='tech_input')
    
    # 1. Decomposizione della Serie
    res_t, trend_t = SeriesDecomposition(kernel_size=25)(in_t)
    
    # Projection base sui residui
    x_t = Dense(128, name='tech_proj')(res_t)
    x_t = PositionalEncoding(max_steps=lookback, max_dims=128)(x_t)
    
    # Block 1
    att_1 = MultiHeadAttention(num_heads=4, key_dim=32, name='mha_res_1')(x_t, x_t)
    x_t = LayerNormalization()(x_t + Dropout(0.2)(att_1))
    x_t = MultiScaleDistillation(filters=128)(x_t)
    
    # Block 2
    att_2 = MultiHeadAttention(num_heads=4, key_dim=32, name='mha_res_2')(x_t, x_t)
    x_t = LayerNormalization()(x_t + Dropout(0.2)(att_2))
    x_t = MultiScaleDistillation(filters=128)(x_t)
    
    x_t_flat = Flatten()(x_t)

    # Elaborazione del Trend
    trend_flat = Flatten()(trend_t)
    trend_feat = Dense(64, activation='gelu')(trend_flat)

    # ── Ramo Macro ──
    in_m = Input(shape=shape_m, name='macro_input')
    x_m = LSTM(64, name='macro_lstm')(in_m)
    x_m = Dropout(0.2, name='macro_drop')(x_m)
    
    # ── Merge ──
    merged = concatenate([x_t_flat, trend_feat, x_m], name='brain_merge')
    x = Dense(512, activation='gelu')(merged) # Aumentata capacità per il multi-task
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x_shared = Dense(128, activation='gelu')(x)
    x_shared = Dropout(0.2)(x_shared)
    
    # ── Dual Head ──
    # 1. Classificazione (0 = Down, 1 = Up)
    dir_out = Dense(1, activation='sigmoid', name='direction_output')(x_shared)
    
    # 2. Regressione (Ampiezza Ritorno)
    mag_out = Dense(1, activation='tanh', name='magnitude_output')(x_shared)
    
    model = Model(inputs=[in_t, in_m], outputs=[dir_out, mag_out], name='Informer_V8_3')
    return model

def get_model(version, weights_path=None, input_shape=None, shape_t=None, shape_m=None, shape_s=None):
    """
    Funzione universale per istanziare e caricare i pesi del modello richiesto.
    """
    if version == "4.3":
        model = build_v4_3_model(input_shape)
    elif version in ["4.6", "crypto_1.7"]:
        model = build_v4_6_model(input_shape)
    elif version in ["5.6", "6.4"]:
        model = build_v5_0_split_brain_model(shape_t, shape_m)
    elif version == "7.0":
        model = build_v7_0_split_brain_model(shape_t, shape_m)
    elif version == "8.0":
        # Lookback preso da shape_t
        lookback = shape_t[0] if shape_t else 60
        model = build_v8_0_informer_model(shape_t, shape_m, lookback=lookback)
    elif version == "8.1":
        lookback = shape_t[0] if shape_t else 60
        model = build_v8_1_informer_model(shape_t, shape_m, lookback=lookback)
    elif version == "8.2":
        lookback = shape_t[0] if shape_t else 60
        model = build_v8_2_informer_model(shape_t, shape_m, lookback=lookback)
    elif version == "8.3":
        lookback = shape_t[0] if shape_t else 60
        model = build_v8_3_informer_model(shape_t, shape_m, lookback=lookback)
    else:
        raise ValueError(f"Versione modello '{version}' non riconosciuta.")
        
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
        
    return model
