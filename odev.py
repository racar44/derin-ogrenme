X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]
w = [-0.7, -0.3]
b = 0.1
lr = 1
epoch = 0

while True:
    epoch += 1
    hata = 0
    for (x1, x2), t in zip(X, y):
        n = w[0]*x1 + w[1]*x2 + b
        o = 1 if n >= 0 else 0
        e = t - o
        w[0] += lr*e*x1
        w[1] += lr*e*x2
        b += lr*e
        hata += abs(e)
        print(f"Epoch {epoch} ({x1},{x2}) t={t} n={round(n, 2)} o={o} e={e} -> w={[round(i, 2) for i in w]} b={round(b, 2)}")
    if hata == 0:
        break

print("Eğitim bitti! Epoch:", epoch)
