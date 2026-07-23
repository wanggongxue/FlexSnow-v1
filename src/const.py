
from pathlib import Path
import numpy as np

def generate_wavelength_um(start=205, stop=5000, step=10):
    return [(start + i * step) * 1E-3 for i in range((stop - start) // step + 1)]

class const:

    shape_types = ["sphere", "ellipsoid", "hexagonal_plate", "Koch_snowflake", "bicontinuous_medium"]
    RvRatio = [1.00, 1.0767, 1.2692, 1.6614, 1.0]

    solar_spectrum = generate_wavelength_um()
    shortwave_spectrum = generate_wavelength_um(start=205, stop=3000, step=10)
    multi_spectrum = generate_wavelength_um(start=305, stop=2500, step=10)

    max_layer = 100
    num_lap = 1 + 2 + 2 + 4 + 1  # 1 water liquid, 2 black carbon types, 2 brown carbon types, 4 dust types, 1 ash type
    density_lap = np.array([1000, 1270, 1657, 1270, 1657, 2645, 2600, 2747, 2000, 2600]) * 1E-3
    rg_c = [6.71627563982869352E-002, 1.48655804E-1]
    rg_d = 0.59594621805492221
    radius_lap = np.array([1.0, rg_c[0], rg_c[1], rg_c[0], rg_c[1], rg_d, rg_d, rg_d, rg_d, rg_d])  # volume-mean radius

    density_ice = 0.917
    density_water = 1.0

    ice_dataset = "ice_Wrn08"
    script_dir = Path(__file__).parent

    rte_path = script_dir / "MLDISORTv2.2"
    lut_path = Path(r"D:\Data\fsnow_optics")
    wet_path = Path(r"D:\Data\fsnow_optics\water_coated_ice")
    lai_path = Path(r"D:\Data\snicar_adv4_OP\lai_in_air")
    algae_path = Path(r"D:\Data\snicar_adv4_OP\sno_alg_pig")

