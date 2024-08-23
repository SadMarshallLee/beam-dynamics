import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy import integrate as integ

# Предварительно вычисляем константы
HC_MeV_mm = 6.582 * 10**(-22) * 2.998 * 10**(11)  # Планк x скорость света в МэВ мм
ME_MeV = 0.511  # масса электрона в МэВ

# Чтение данных из файла
data = np.loadtxt('Efield.txt')
z, Ez = data[:, 0], data[:, 1]

# Основные параметры для расчета импульса и энергии
c = const.c
e = const.e
phi = np.pi / 2
W_MeV = 4.6e-6
f = 2856e6  # частота
t = 350e-12  # время

# Функция для расчета продольного импульса и скорости
def longitudinal_momentum(W_MeV, Me_MeV, c):
    if W_MeV <= Me_MeV:
        pz_MeV = np.sqrt(2 * Me_MeV * W_MeV)
        v = pz_MeV / (Me_MeV * c)
    elif W_MeV <= 10 * Me_MeV:
        under_sqrt = W_MeV**2 - Me_MeV**2
        pz_MeV = np.sqrt(under_sqrt) if under_sqrt >= 0 else np.nan
        v = np.sqrt(1 - (Me_MeV / W_MeV)**2) * c if under_sqrt >= 0 else np.nan
    else:
        pz_MeV = np.sqrt(W_MeV**2 - Me_MeV**2)
        v = c
    return pz_MeV, v

# Функция для определения значения фундаментальной моды
def fundamental_mode(n, l):
    return 2 * n + l + 1

# Функция для расчета поперечного импульса в МэВ, на вход размер пакета в мм
def transverse_momentum(r0):
    return HC_MeV_mm / r0

# Длина волны де Бройля
def de_broglie_wavelength(pz):
    return HC_MeV_mm / pz if pz > 0 else np.nan

# Средний начальный размер пакета, мм
def rho_mid(r0, M):
    return r0 * np.sqrt(M)

# Угол раскрытия электрона
def theta(pt, pz):
    return pt / pz * 180 / np.pi if pz > 0 else np.nan

# Расплывание в ближней зоне
def fresnel_zone_packet_width(z, ldb, rho_mid0, M):
    if z == 0:
        return rho_mid0
    term = (2 * np.pi * rho_mid0**2) / (z * ldb)
    return ((z * ldb) / (2 * np.pi * rho_mid0)) * M * (1 + 0.5 * (term / M)**2)

# Расплывание в дальней зоне
def far_field_packet_width(z, ldb, rho_mid, M):
    return (z / rho_mid) * M * (ldb / (2 * np.pi))

# Длина когерентности
def coherence_length(r_mid):
    return np.sqrt(2) * r_mid

# Расчет ширины пакета для заданной длины
def real_packet_width(z_values, W_MeV, r0, n, l):
    M = fundamental_mode(n, l)
    pz, _ = longitudinal_momentum(W_MeV, ME_MeV, c)
    ldb = de_broglie_wavelength(pz)
    rho_mid0 = rho_mid(r0, M)
    ksi = coherence_length(rho_mid0)  # начальная длина когерентности

    width = []
    for z in z_values:
        if z <= ksi:
            width.append(fresnel_zone_packet_width(z, ldb, rho_mid0, M))
        else:
            width.append(far_field_packet_width(z, ldb, rho_mid0, M))
    
    return width

# Расчет ширины пакета для заданной длины с учетом изменения pz  
def real_packet_width_with_pz(z_values, pz_profile, v_profile, r0, n, l):
    M = fundamental_mode(n, l)

    # Интегрируем профиль скорости по z для создания new_z
    new_z = np.zeros_like(z_values)
    for i in range(1, len(z_values)):
        new_z[i] = integ.simps(v_profile[:i+1], z_values[:i+1])
    
    # Переводим pz из МэВ в эВ
    pz_profile_eV = pz_profile * 1e6

    print("Значения new_z (в метрах):", new_z)
    ldb_values = [de_broglie_wavelength(pz) for pz in pz_profile_eV]
    rho_mid0 = rho_mid(r0, M)
    ksi = coherence_length(rho_mid0)  # начальная длина когерентности

    width_with_pz = []
    for z, ldb in zip(new_z, ldb_values):
        if np.isnan(ldb):
            width_with_pz.append(np.nan)
        elif z <= ksi:
            width_with_pz.append(fresnel_zone_packet_width(z, ldb, rho_mid0, M))
        else:
            width_with_pz.append(far_field_packet_width(z, ldb, rho_mid0, M))
    
    return width_with_pz

# Расчет расстояния для заданной ширины пакета
def distance_for_a_width(r0, r_fin, energy, n, l):
    M = fundamental_mode(n, l)
    pz, _ = longitudinal_momentum(energy, ME_MeV, const.c)
    ldb = de_broglie_wavelength(pz)
    rho_mid0 = rho_mid(r0, M)
    return (2 * np.pi * r_fin * rho_mid0) / (ldb * M)
   
# Функция для построения общей картинки со всеми графиками (добавлен новый график)
def plot_all_graphs(z, r_mid, energy_profile, pz_profile, r_mid_with_pz, v_profile):
    plt.figure(figsize=(12, 10))

    # График 1: Размеры волнового пакета вдоль оси z
    plt.subplot(2, 3, 1)
    plt.plot(z, r_mid, label="Размеры волнового пакета вдоль оси z")
    plt.xlabel("Длина распространения, м")
    plt.ylabel("Ширина пакета, мм")
    plt.legend()

    # График 2: Энергия вдоль оси z
    plt.subplot(2, 3, 2)
    plt.plot(z, energy_profile, label="Энергия вдоль оси z")
    plt.xlabel('z (м)')
    plt.ylabel('Энергия (МэВ)')
    plt.legend()

    # График 3: Продольный импульс (pz)
    plt.subplot(2, 3, 3)
    plt.plot(z, pz_profile, label="Продольный импульс (pz)")
    plt.xlabel('z (м)')
    plt.ylabel('pz (МэВ/c)')
    plt.legend()

    # График 4: Ширина пакета с учетом изменения pz
    plt.subplot(2, 3, 4)
    plt.plot(z, r_mid_with_pz, label="Ширина пакета с учетом изменения pz")
    plt.xlabel("Длина распространения, мм")
    plt.ylabel("Ширина пакета, мм")
    plt.legend()

    # График 5: Скорость вдоль оси z
    plt.subplot(2, 3, 5)
    plt.plot(z, v_profile, label="Скорость вдоль оси z")
    plt.xlabel('z (м)')
    plt.ylabel('Скорость (м/с)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

# Основной блок
r0 = 1e-9  # размер пакета в м
n = 3
l = 1
step_of_z = (z[1] - z[0])
length = z[-1]

# Расчет импульса и скорости
pz_MeV, v = longitudinal_momentum(W_MeV, ME_MeV, c)

if np.isnan(v) or v == 0:
    print("Ошибка: скорость v равна NaN или нулю.")
    dE = energy_change_profile = pz_profile = v_profile = np.nan * np.ones_like(z)
else:
    denominator = f * np.mean(z) / (2 * v) if v != 0 else 1
    T = np.sin(denominator) / denominator if denominator != 0 else 0
    
    try:
        integral_result = integ.simps(Ez, z)
        dE = e * integral_result * T * np.cos(f * t + phi)
        
        delta_z = (z[1] - z[0])
        energy_profile = np.cumsum(Ez * delta_z) + W_MeV
        
        energy_change_profile = energy_profile - W_MeV + dE
        pz_profile = np.array([longitudinal_momentum(E, ME_MeV, c)[0] for E in energy_profile])
        v_profile = np.array([longitudinal_momentum(E, ME_MeV, c)[1] for E in energy_profile])
        
    except Exception as ex:
        print(f"Ошибка при вычислении интеграла: {ex}")
        dE = energy_change_profile = pz_profile = v_profile = np.nan * np.ones_like(z)

r_mid = real_packet_width(z, W_MeV, r0, n, l)
r_mid_with_pz = real_packet_width_with_pz(z, pz_profile, v_profile, r0, n, l)

# Построение общей картинки со всеми графиками (добавлен v_profile)
plot_all_graphs(z, r_mid, energy_profile, pz_profile, r_mid_with_pz, v_profile)