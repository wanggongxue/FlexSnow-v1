
from src import *
from pathlib import Path
import math

def generate_wavelength_um(start=205, stop=5000, step=10):
    return [(start + i * step) * 1E-3 for i in range((stop - start) // step + 1)]

def main():
    """
    to run the FlexSnow for snow externally mixing with LACs
    """

    """
    working directories
    """
    work_dir = Path(r"E:\WORK\FsnowV2_2026\Test\LAC_ext")
    tmp_dir = work_dir / "tmp"

    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    """
    directories of optical library
    """
    ice_dataset = const.ice_dataset
    rte_path = const.rte_path
    lut_path = const.lut_path
    lai_path = const.lai_path

    """
    parameters to calculate snow single-scattering properties
    """
    # number of snow layers
    nlyr = 2

    #  shape id for all layers
    shp_idx = [0, 1]
    shape_types = ["sphere", "ellipsoid", "hexagonal_plate", "Koch_snowflake", "bicontinuous_medium"]
    # ice particle radius in um
    ice_radius = [50,60]
    # layer thickness in cm
    thickness = [0.5,100]
    # layer density in g/cm3
    density = [0.25,0.25]
    # LAC mass concentration for all layers
    mass_conc = [1e-6,0.0] #1e-6 1 ppm,  1e-9 1ppb
    # LAC files
    file_LAC = ["bc_ChCB_rn40_dns1270.nc","bc_ChCB_rn40_dns1270.nc"]

    """
    parameters to run the disort
    """
    # negative solar zenith angle indicates diffuse irradiannce
    ang_res = 5
    zenith = [-1] + [i for i in range(0,91,ang_res)]
    if math.isclose(zenith[-1], 90.0):
        zenith[-1] = 89.99
    # you may want loop over wavelengths
    wavelengths = const.multi_spectrum

    smooth_phase = True  # if you want "original" phase function, turn it off
    NMOM = 1000
    NSTR = 16
    append = True


    '''
    the following code calculates snow single-scattering properties
    '''

    m = snow_ssp(num_layer=nlyr,
                 num_nonice=1,
                 max_moment=NMOM,
                 smooth_phase=smooth_phase,
                 output_path=tmp_dir)
    for j in range(nlyr):
        if shp_idx[j] == 4:
            m.ice_file[j] = lut_path/ ice_dataset/ f"{shape_types[shp_idx[j]]}_{ice_radius[j]}um.nc"
        else:
            m.ice_file[j] = lut_path/ice_dataset/f"{shape_types[shp_idx[j]]}_rough_{ice_radius[j]}um.nc"
        m.lap_file[j, 0] = lai_path / file_LAC[j]
        m.lap_mass_fraction[j, 0] = mass_conc[j]
    m.run()

    fileAlbedo = work_dir / f"DHR.bin"
    fileBRF = work_dir / f"BRF.bin"
    fileAlbedo.unlink(missing_ok=True)
    fileBRF.unlink(missing_ok=True)
    n = 0
    for sza in zenith:
        for wavelength in wavelengths:
            n = n + 1
            afn = [tmp_dir / f"layer={layer}_wvl={wavelength:.3f}um.txt" for layer in range(nlyr)]

            flag = f"wvl={wavelength:.3f}um_sza={sza:.1f}.txt"

            if sza < 0:
                irrDir = 0.0
                NPHI = int(180/ang_res+1) #设置为1是因为漫入射天空光照射条件下，反射光对方位角无响应
                NUMU = int(90/ang_res+1)
            else:
                irrDir = 1.0
                NPHI = int(180/ang_res+1) # 注意要是单数，181意味着角度分辨率是1度，如果不画极坐标图，可以设置为3，适合画主平面截平面曲线图
                NUMU = int(90/ang_res+1)
            if n==1:
                append_tf = False
            else:
                append_tf = append
            instance = MLDISORTv2(sza=sza, irrDir=irrDir, NLYR=nlyr, NPHI=NPHI, NUMU=NUMU,
                                  NSTR=NSTR, NMOM=NMOM, density=density, depth=thickness,
                                  append=append_tf)
            instance._exefile = rte_path
            instance._inputfile = tmp_dir / ("input_" + flag)
            instance._logfile = tmp_dir / ("log_" + flag)
            instance.afn = afn
            instance.fileAlbedo = fileAlbedo
            instance.fileBRF = fileBRF

            instance.run()
            print(instance._logfile)
            print()
if __name__ == "__main__":
    print()
    main()