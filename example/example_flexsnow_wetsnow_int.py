
from src import *
from pathlib import Path
def generate_wavelength_um(start=205, stop=5000, step=10):
    return [(start + i * step) * 1E-3 for i in range((stop - start) // step + 1)]

def main():
    """
    to run the FlexSnow for snow externally mixing with LACs
    """

    """
    working directories
    """
    work_dir = Path(r"E:\WORK\FsnowV2_2026\Test\WetSnow_int")
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
    wet_path = const.wet_path

    """
    parameters to calculate snow single-scattering properties
    """
    # number of snow layers
    nlyr = 2
    # volume percentage of liquid water
    fv = [10,10]
    # snow grain radius in um
    radius = [500,600]
    # layer thickness in cm
    thickness = [0.5,100]
    # layer density in g/cm3
    density = [0.25,0.25]

    """
    parameters to run the disort
    """
    # you may want loop over solar zenith angles
    # negative solar zenith angle indicates diffuse irradiannce
    zenith = [-1, 40, 50]
    # you may want loop over wavelengths
    wavelengths = const.multi_spectrum
    smooth_phase = True # if you want "original" phase function, turn it off
    NMOM = 1000
    NSTR = 16

    '''
    the following code calculates snow single-scattering properties
    '''

    m = snow_ssp(num_layer=nlyr,
                 max_moment=NMOM,
                 smooth_phase=smooth_phase,
                 output_path=tmp_dir)
    for j in range(nlyr):
        m.ice_file[j] = lut_path/ wet_path/ f"IceCoatedByWater_re{radius[j]}um_fv{fv[j]}%.nc"
    m.run()

    for sza in zenith:
        for wavelength in wavelengths:
            afn = [tmp_dir / f"layer={layer}_wvl={wavelength:.3f}um.txt" for layer in range(nlyr)]

            flag = f"wvl={wavelength:.3f}um_sza={sza:.1f}.txt"
            fileAlbedo = work_dir / ("DHR_" + flag)
            fileBRF = work_dir / ("BRF_" + flag)

            if sza < 0:
                irrDir = 0.0
                NPHI = 1
                NUMU = 1
            else:
                irrDir = 1.0
                NPHI = 3
                NUMU = 91
            instance = MLDISORTv2(sza=sza, irrDir=irrDir, NLYR=nlyr, NPHI=NPHI, NUMU=NUMU,
                                  NSTR=NSTR, NMOM=NMOM, density=density, depth=thickness)
            instance._exefile = rte_path
            instance._inputfile = tmp_dir / ("input_" + flag)
            instance._logfile = tmp_dir / ("log_" + flag)
            instance.afn = afn
            instance.fileAlbedo = fileAlbedo
            instance.fileBRF = fileBRF

            instance.run()
            print(instance.fileAlbedo)
            print()
if __name__ == "__main__":
    print()
    main()