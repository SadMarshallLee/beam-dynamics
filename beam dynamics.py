import numpy as np
import scipy.constants as const
from scipy import integrate as integ
import matplotlib.pyplot as plt

def longitudinal_momentum(W_MeV, Me_MeV, me, c):
    if W_MeV <= Me_MeV:
        pz_MeV = np.sqrt(2 * me * W_MeV * 1e6 / const.e) * 1e-6 * const.e
        v = pz_MeV / (me * c) * 1e6 / const.e
    elif W_MeV <= 10 * Me_MeV:
        under_sqrt = W_MeV**2 - Me_MeV**2
        pz_MeV = np.sqrt(under_sqrt) if under_sqrt >= 0 else np.nan
        v = np.sqrt(1 - (Me_MeV / W_MeV)**2) * c
    else:
        pz_MeV = W_MeV
        v = c
    return pz_MeV, v


# Чтение данных из файла
data = np.loadtxt('Efield.txt')
z, Ez = data[:, 0], data[:, 1]

# Основные параметры
c = const.c
E0 = np.mean(Ez)
e = const.e
phi = np.pi / 2
W_ev = 4.6e-6
me = const.electron_mass
Me_MeV = 0.511
f = 2856e6
t = 10e-12

# Расчет импульса и энергии
pz_MeV, v = longitudinal_momentum(W_ev, Me_MeV, me, c)

if np.isnan(v) or v == 0:
    print("Ошибка: скорость v равна NaN или нулю.")
    dE = energy_change_profile = pz_profile = np.nan * np.ones_like(z)
else:
    denominator = f * np.mean(z) / (2 * v) if v != 0 else 1
    T = np.sin(denominator) / denominator if denominator != 0 else 0
    
    try:
        integral_result = integ.simps(Ez, z)
        dE = e * integral_result * T * np.cos(f * t + phi)
        
        delta_z = z[1] - z[0]
        energy_profile = np.cumsum(Ez * delta_z) + W_ev
        
        energy_change_profile = energy_profile - W_ev + dE
        pz_profile = np.array([longitudinal_momentum(E, Me_MeV, me, c)[0] for E in energy_profile])
        
    except Exception as ex:
        print(f"Ошибка при вычислении интеграла: {ex}")
        dE = energy_change_profile = pz_profile = np.nan * np.ones_like(z)

print(f"pz (в МэВ): {pz_MeV}, dE: {dE}, T: {T}, Energy: {energy_profile[-1]}")

# Построение графиков
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(z, Ez, label='Ez')
plt.xlabel('z')
plt.ylabel('Ez')
plt.title('Электрическое поле вдоль координаты z')

plt.subplot(2, 2, 2)
plt.plot(z, energy_change_profile, label='Изменение энергии', color='orange')
plt.xlabel('z')
plt.ylabel('Изменение энергии (МэВ)')
plt.title('Изменение энергии относительно начальной энергии')

plt.subplot(2, 2, 3)
plt.plot(z, pz_profile, label='Импульс', color='green')
plt.xlabel('z')
plt.ylabel('Импульс (МэВ)')
plt.title('Изменение импульса вдоль оси z')

plt.tight_layout()
plt.show()