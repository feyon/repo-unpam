import matplotlib.pyplot as plt

# Data untuk grafik
x = [1, 2, 3, 4, 5, 6]
membership_values = [0.9, 0.8, 0.3, 0.2, 0.7, 0.1]

# Membuat grafik
plt.figure(figsize=(8, 5))
plt.plot(x, membership_values, marker='o', linestyle='-', color='blue')
plt.title('Grafik Himpunan Fuzzy A')
plt.xlabel('Elemen X')
plt.ylabel('Derajat Keanggotaan (μ_A(x))')
plt.ylim(0, 1.1)
plt.grid(True)
plt.xticks(x)
plt.show()

