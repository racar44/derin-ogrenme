import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error, r2_score

# Okul numarasinin son 2 hanesi (363425005 -> 05)
SEED = 5
EPOCHS = 300

os.makedirs("odev_grafikleri", exist_ok=True)

# Veri seti: t = 1 + sin(pi * p / 4), -2 <= p <= 2
p = np.linspace(-2, 2, 500).reshape(-1, 1)
y_true = 1 + np.sin(np.pi * p / 4)


def deney(aktivasyon="tanh", lr=0.1, opt_adi="sgd", dagilim="normal",
          katman=1, noron=2):
    # Her deneyden once ayni baslangic kosullari
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = Sequential()
    model.add(Input(shape=(1,)))
    for _ in range(katman):
        model.add(Dense(noron, activation=aktivasyon, kernel_initializer=dagilim))
    model.add(Dense(1, activation="linear", kernel_initializer=dagilim))

    optimizatorler = {
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adam": tf.keras.optimizers.Adam,
    }
    model.compile(optimizer=optimizatorler[opt_adi](learning_rate=lr), loss="mse")

    t0 = time.time()
    history = model.fit(p, y_true, epochs=EPOCHS, verbose=0)
    sure = time.time() - t0

    y_pred = model.predict(p, verbose=0)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
        "sure": sure,
        "loss": history.history["loss"],
        "pred": y_pred,
    }


def grafik_ciz(baslik, dosya, s):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    ax[0].plot(s["loss"], color="firebrick", linewidth=2)
    ax[0].set_yscale("log")
    ax[0].set_title(f"{baslik}: Kayip (Logaritmik)")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("MSE")
    ax[0].grid(True, which="both", linestyle="--", alpha=0.4)

    ax[1].plot(p, y_true, label="Hedef", color="#1f4e79", linewidth=2)
    ax[1].plot(p, s["pred"], label="Ag cikisi", color="firebrick", linewidth=2)
    ax[1].set_title(f"{baslik}: Hedef ve Ag Cikisi")
    ax[1].set_xlabel("p (giris)")
    ax[1].set_ylabel("cikis")
    ax[1].legend()
    ax[1].grid(True, linestyle="--", alpha=0.4)

    ax[2].scatter(y_true, s["pred"], alpha=0.5, color="firebrick", s=15)
    ax[2].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
               "r--", linewidth=1.5)
    ax[2].set_title(f"{baslik}: R2 Grafigi (R2 = {s['r2']:.4f})")
    ax[2].set_xlabel("Gercek Degerler (Hedef)")
    ax[2].set_ylabel("Tahmin Edilen Degerler")
    ax[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"odev_grafikleri/{dosya}.png", dpi=300)
    plt.close()


def tablo_calistir(no, baslik, secenekler, **sabit):
    """secenekler: {etiket: {degisen parametre}} . En iyi satir: R2 en yuksek, esitse MSE en dusuk."""
    print(f"\n=== TABLO {no}: {baslik} ===")
    sonuclar = {}
    for etiket, degisen in secenekler.items():
        sonuclar[etiket] = deney(**{**sabit, **degisen})

    en_iyi = max(sonuclar, key=lambda k: (round(sonuclar[k]["r2"], 6), -sonuclar[k]["mse"]))

    print(f"{'Parametre':<22}{'MSE':>11}{'RMSE':>11}{'R2':>11}{'Sure(sn)':>11}")
    for etiket, s in sonuclar.items():
        isaret = "  <-- EN IYI" if etiket == en_iyi else ""
        print(f"{etiket:<22}{s['mse']:>11.6f}{s['rmse']:>11.6f}{s['r2']:>11.6f}{s['sure']:>11.2f}{isaret}")

    dosya = f"tablo{no}_{str(en_iyi).lower().replace(' ', '_').replace('.', '_')}"
    grafik_ciz(f"Tablo{no}_{en_iyi}", dosya, sonuclar[en_iyi])
    print(f"Kazanan grafik kaydedildi: odev_grafikleri/{dosya}.png")


# Tablo 1: aktivasyon (lr=0.1, sgd, normal, 1 katman, 2 noron)
tablo_calistir(1, "AKTIVASYON", {
    "Lineer": {"aktivasyon": "linear"},
    "Sigmoid": {"aktivasyon": "sigmoid"},
    "Hiperbolik Tanjant": {"aktivasyon": "tanh"},
    "ReLU": {"aktivasyon": "relu"},
    "LeakyReLU": {"aktivasyon": "leaky_relu"},
    "ELU": {"aktivasyon": "elu"},
    "SiLU": {"aktivasyon": "swish"},
}, lr=0.1, opt_adi="sgd", dagilim="normal", katman=1, noron=2)

# Tablo 2: ogrenme orani (tanh)
tablo_calistir(2, "OGRENME ORANI", {
    "0.1": {"lr": 0.1}, "0.01": {"lr": 0.01}, "0.001": {"lr": 0.001},
}, aktivasyon="tanh", opt_adi="sgd", dagilim="normal", katman=1, noron=2)

# Tablo 3: optimizator (tanh, lr=0.1)
tablo_calistir(3, "OPTIMIZATOR", {
    "SGD": {"opt_adi": "sgd"}, "RMSprop": {"opt_adi": "rmsprop"}, "Adam": {"opt_adi": "adam"},
}, aktivasyon="tanh", lr=0.1, dagilim="normal", katman=1, noron=2)

# Tablo 4: baslangic dagilimi
tablo_calistir(4, "BASLANGIC DAGILIMI", {
    "normal": {"dagilim": "normal"}, "uniform": {"dagilim": "uniform"},
}, aktivasyon="tanh", lr=0.1, opt_adi="sgd", katman=1, noron=2)

# Tablo 5: gizli katman sayisi (her katmanda 2 noron)
tablo_calistir(5, "KATMAN SAYISI", {
    str(k): {"katman": k} for k in [1, 2, 5, 10, 20]
}, aktivasyon="tanh", lr=0.1, opt_adi="sgd", dagilim="normal", noron=2)

# Tablo 6: noron sayisi (tek gizli katman)
tablo_calistir(6, "NORON SAYISI", {
    str(n): {"noron": n} for n in [2, 5, 10, 20, 50]
}, aktivasyon="tanh", lr=0.1, opt_adi="sgd", dagilim="normal", katman=1)

print("\nBitti. Kazanan grafikler 'odev_grafikleri' klasorunde.")
