
import os
import subprocess
from pathlib import Path
from typing import Union, List
import numpy as np

class MLDISORT:
    def __init__(
        self,
        sza: float = 50.0,
        irrDir: float = 1.0,
        loadAng: bool = False,
        NUMU: int = 91,
        NPHI: int = 181,
        iUMU: Union[float, List[float]] = 0.0,
        iPHI: Union[float, List[float]] = 0.0,
        PHI0: float = 0.0,
        ALBEDO: float = 0.0,
        NSTR: int = 32,
        BRDF_TYPE: int = 0,
        BRDF_ARG: List[float] = [0.0] * 4,
        NMOM: int = 10000,
        fileAlbedo:str = "albedo.txt",
        fileBRF:str = "reflectance.txt",
        fileGEO:str = "input_geometry.txt",
        geometry=np.array([0, 0, 0, 0]),
    ):
        """初始化所有属性 / Initialize all attributes"""
        self.sza = sza
        self.irrDir = irrDir
        self.loadAng = loadAng
        self.NUMU = NUMU
        self.NPHI = NPHI
        self.iUMU = iUMU
        self.iPHI = iPHI
        self.PHI0 = PHI0
        self.ALBEDO = ALBEDO
        self.NSTR = NSTR
        self.BRDF_TYPE = BRDF_TYPE
        self.BRDF_ARG = BRDF_ARG if BRDF_ARG is not None else [0.0] * 4
        self.NMOM = NMOM
        self.fileAlbedo = fileAlbedo
        self.fileBRF = fileBRF
        self.fileGEO = fileGEO
        self.geometry = geometry
        self._exefile = "MLDISORTv2.exe"
        self._inputfile = "inputDISORT.txt"
        self._logfile = 'log.txt'
        self.iUMU = self._ensure_list(self.iUMU, float)
        self.iPHI = self._ensure_list(self.iPHI, float)

        # 验证 / Validate
        if len(self.BRDF_ARG) != 4:
            raise ValueError("BRDF_ARG必须包含4个元素 / BRDF_ARG must have exactly 4 elements")

    @staticmethod
    def _ensure_list(value: Union[float, List[float]], dtype: type) -> List[float]:
        """确保输入为列表 / Ensure input is a list"""
        return [dtype(value)] if isinstance(value, (int, float)) else [dtype(v) for v in value]
    def run(self):
        if not os.path.exists(self._exefile):
            raise FileNotFoundError(f"Executable file {self._exefile} not found.")

        self._write_to_file()

        try:
            logfile = open(self._logfile, "w") if self._logfile else subprocess.DEVNULL
            inputfile_path = Path(self._inputfile).resolve()
            result = subprocess.run(
                [self._exefile, inputfile_path],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                text=True
            )
        except Exception as e:
            raise RuntimeError(f"Error running {self._exefile}: {e}")
        finally:
            if self._logfile:
                logfile.close()

    def __str__(self):
        """根据用户初始化结果生成字符串表示 / Generate string representation based on initialization"""
        attributes = vars(self)
        result = ["MLDISORT configuration:"]
        for key, value in attributes.items():
            if key.startswith("_"):
                continue
            result.append(f"  {key} = {value}")
        return "\n".join(result)

    def write_angles_file(self):
        angles = np.array(self.geometry)
        if angles.ndim != 2 or angles.shape[1] != 4:
            raise ValueError("angles must be a 2D array with 4 columns")

        with open(self.fileGEO, 'w') as f:
            f.write("! line 1: nangles. lines 2-(nangles+1): solar zenith, solar azimuth, view zenith, view azimuth\n")
            f.write(f"{angles.shape[0]}\n")
            for row in angles:
                f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")


    def _write_to_file(self):
        with open(self._inputfile, "w") as f:
            f.write("&input\n\n")
            f.write(f"sza = {self.sza}\n")
            f.write(f"irrDir = {self.irrDir}\n")
            f.write(f"loadAng = {'.True.' if self.loadAng else '.False.'}\n")
            if hasattr(self, 'loadPhase'):
                f.write(f"loadPhase = {'.True.' if self.loadPhase else '.False.'}\n")
            if hasattr(self, 'append'):
                f.write(f"append = {'.True.' if self.append else '.False.'}\n")
            f.write("\n")

            if hasattr(self, 'afn') and hasattr(self, 'NLYR'):
                f.write(f"NLYR = {self.NLYR}\n")
                if isinstance(self.afn, list):
                    for j in range(self.NLYR):
                        f.write(f"afn({j + 1})(:) = '{self.afn[j]}'\n")
                else:
                    for j in range(self.NLYR):
                        f.write(f"afn({j + 1})(:) = '{self.afn}'\n")

            if hasattr(self, 'defTAU'):
                if self.defTAU[0]<0:
                    if hasattr(self, 'density'):
                        if isinstance(self.density, list):
                            values = ",".join(str(v) for v in self.density)
                        else:
                            values = str(self.density)
                        f.write(f"density = {values}\n")

                    if hasattr(self, 'depth'):
                        if isinstance(self.depth, list):
                            values = ",".join(str(v) for v in self.depth)
                        else:
                            values = str(self.depth)
                        f.write(f"depth = {values}\n")


            if hasattr(self, 'defTAU'):
                if self.defTAU[0] > 0:
                    if isinstance(self.defTAU, list):
                        values = ",".join(str(v) for v in self.defTAU)
                    else:
                        values = str(self.defTAU)
                    f.write(f"defTAU = {values}\n")

            f.write(f"NUMU = {int(self.NUMU)}\n")
            f.write(f"NPHI = {int(self.NPHI)}\n")

            if isinstance(self.iUMU, list):
                values = ",".join(str(v) for v in self.iUMU)
            else:
                values = str(self.iUMU)
            f.write(f"iUMU = {values}\n")

            if isinstance(self.iPHI, list):
                values = ",".join(str(v) for v in self.iPHI)
            else:
                values = str(self.iPHI)
            f.write(f"iPHI = {values}\n")

            f.write(f"PHI0 = {int(self.PHI0)}\n")

            f.write("\n")

            f.write(f"ALBEDO = {self.ALBEDO}\n")
            f.write(f"BRDF_TYPE = {self.BRDF_TYPE}\n")
            f.write(f"BRDF_ARG = {','.join(str(v) for v in self.BRDF_ARG)}\n")

            f.write("\n")

            f.write(f"NSTR = {int(self.NSTR)}\n")
            f.write(f"NMOM = {int(self.NMOM)}\n")


            f.write("\n")
            if hasattr(self,'IPHAS'):
                f.write(f"IPHAS = {self.IPHAS}\n")

            if hasattr(self, 'fileSSP'):
                f.write(f"fileSSP = '{self.fileSSP}'\n")
            f.write(f"fileAlbedo = '{self.fileAlbedo}'\n")
            f.write(f"fileBRF = '{self.fileBRF}'\n")
            if hasattr(self, 'fileGEO'):
                f.write(f"fileGEO = '{self.fileGEO}'\n")
            if hasattr(self, 'fileExpan'):
                f.write(f"fileExpan = '{self.fileExpan}'\n")

            f.write("\n")
            f.write("\n/")


class MLDISORTv1(MLDISORT):
    def __init__( self, sza=50.0, irrDir=1.0, loadAng=False, NUMU=91, NPHI=181,
        iUMU=0.0, iPHI=0.0, PHI0=0.0, ALBEDO=0.0, NSTR=32,
        BRDF_TYPE=0, BRDF_ARG=None, NMOM=10000, fileAlbedo="albedo.txt",
        fileBRF="reflectance.txt", fileGEO="input_geometry.txt",geometry=np.array([0,0,0,0]),
        fileSSP:str= "input_ssp.txt" ):

        super().__init__(sza, irrDir, loadAng,  NUMU, NPHI, iUMU, iPHI, PHI0, ALBEDO, NSTR, BRDF_TYPE, BRDF_ARG, NMOM,
                  fileAlbedo, fileBRF, fileGEO,geometry)
        self.fileSSP = fileSSP
        self._exefile = "MLDISORTv1.exe"



class MLDISORTv2(MLDISORT):
    def __init__( self,sza=60.0, irrDir=1.0, loadAng=False, NUMU=91, NPHI=181,
        iUMU=0.0, iPHI=0.0, PHI0=0.0, ALBEDO=0.0, NSTR=16,
        BRDF_TYPE=0, BRDF_ARG=None, NMOM=1000, IPHAS=0,fileAlbedo="albedo.txt",
        fileBRF="reflectance.txt", fileGEO="input_geometry.txt",geometry=np.array([0,0,0,0]),
        loadPhase: bool = True,append:bool=False,NLYR: int = 1,
        afn: Union[str, List[str]] = "shape=01_LAP=None_SSP.txt",
        density: Union[float, List[float]] = 0.25,
        depth: Union[float, List[float]] = 100.0,
        defTAU: Union[float, List[float]] = -1.0):

        super().__init__(sza, irrDir, loadAng,  NUMU, NPHI, iUMU, iPHI, PHI0, ALBEDO, NSTR, BRDF_TYPE, BRDF_ARG, NMOM,
                  fileAlbedo, fileBRF, fileGEO,geometry)
        self.loadPhase = loadPhase
        self.append = append
        self.NLYR = NLYR
        self.afn = afn
        self.density = density
        self.depth = depth
        self.defTAU = defTAU
        self.IPHAS=IPHAS
        self._process_array_attr("afn", str)
        self._process_array_attr("density", float)
        self._process_array_attr("depth", float)
        self._process_array_attr("defTAU", float)

    def _process_array_attr(self, attr: str, dtype: type):
        """处理数组属性：标量转列表 / Process array attribute: convert scalar to list"""
        value = getattr(self, attr)
        if isinstance(value, (int, float, str)):
            setattr(self, attr, [dtype(value)] * self.NLYR)
        elif len(value) != self.NLYR:
            raise ValueError(f"{attr}长度必须与NLYR({self.NLYR})一致 / {attr} length must match NLYR")




