
import netCDF4 as nc
import os
from src.const import const
from src.He_scheme import *
from src.Legendre_expansion import *

class snow_ssp:
    ct = const()
    def __init__(self,num_layer=1,num_nonice=0,remove_diffraction=False,truncate_forward=False,
                 smooth_phase=False,use_HG_phase=False,max_moment = 1000,output_path="c:/"):
        self._num_layer = None
        self.num_layer = num_layer
        self._num_nonice = None
        self.num_nonice = num_nonice
        self.remove_diffraction = remove_diffraction
        self.truncate_forward = truncate_forward
        self.max_moment = max_moment
        self.smooth_phase = smooth_phase
        self.use_HG_phase=use_HG_phase
        self.output_path = output_path
        self._init_para()
    def _init_para(self):
        nly = self._num_layer
        nni = self._num_nonice
        if nni>=1:
            size = (nly,nni)
        else:
            size = (nly,)
        self.ice_file = np.array(["sphere_rough_100um.nc"] * nly, dtype=object)
        self.lap_file = np.full(size, "bc_ChCB_rn40_dns1270.nc", dtype=object)
        self.ice_density = np.array([0.917] * nly)
        self.lap_density = np.full(size, [0.0])
        self.ice_volume_radius = np.array([100] * nly)
        self.lap_volume_radius =  np.full(size, [0.04])
        self.lap_mass_fraction = np.full(size, [0.0])
        self.soot_int_mass_fract = np.zeros(nly)
        self.dust_int_mass_fract = np.zeros(nly)
        self.dust_ext_mass_fract = np.zeros(nly)

    def run(self):
        nni = self._num_nonice
        max_moment = self.max_moment
        x, w = gauss_zeroes_weights(-1, 1, max_moment)
        Pl = legendre_polynomial(x, max_moment)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        for i in range(self._num_layer):
            (wvl, omega, g,  Mext,
             nAngle, nPhase, idGauss, theta, dmu, F11) = readIceSSP(self.ice_file[i],self.remove_diffraction,self.truncate_forward,self.use_HG_phase)
            mu = np.cos(theta * np.pi / 180.0)
            if self.remove_diffraction:
                add_factor = 1.0
                mul_factor = 0.5
            else:
                add_factor = 0.0
                mul_factor = 1.0
            omega_in = mul_factor * (omega + add_factor)
            if self.soot_int_mass_fract[i] > 0.0:
                omega_out = soot_lower_omega(omega_in, wvl, self.soot_int_mass_fract[i] * 1E9, self.ct.density_lap[1],
                                            self.ct.radius_lap[1])
            elif self.dust_int_mass_fract[i] > 0.0:
                omega_out = dust_lower_omega(omega_in, wvl, self.dust_int_mass_fract[i] * 1E9, 1)
            elif self.dust_ext_mass_fract[i] > 0.0:
                omega_out = dust_lower_omega(omega_in, wvl, self.dust_ext_mass_fract[i] * 1E9, 0)
            else:
                omega_out = omega_in
            omega = omega_out / mul_factor - add_factor

            if nni>= 1:
                wgt = np.insert(self.lap_mass_fraction[i, :], 0, 1 - sum(self.lap_mass_fraction[i, :]))
                Msca = Mext * omega
                Msca_up = wgt[0] * Msca
                Mext_up = wgt[0] * Mext
                g_up = wgt[0] * Msca * g

                p11_up = wgt[0] * Msca.reshape(-1, 1) * F11
                nwvl, = Msca.shape
                for j in range(self._num_nonice):
                    if self.lap_mass_fraction[i, j] > 0.0:
                        (_, omega_lap, g_lap, Mext_lap,particle_density,
                         _, _, _, _, F11_lap) = read_lai(self.lap_file[i, j])
                        if self.lap_density[i,j]>0:
                            Mext_lap = Mext_lap * particle_density / self.lap_density[i,j]
                        Msca_lap = Mext_lap[0:nwvl] * omega_lap[0:nwvl]
                        Msca_up += wgt[j + 1] * Msca_lap
                        Mext_up += wgt[j + 1] * Mext_lap[0:nwvl]
                        g_up += wgt[j + 1] * Msca_lap*g_lap[0:nwvl]
                        p11_up += wgt[j + 1] * Msca_lap.reshape(-1, 1) * F11_lap[0:nwvl,:]

                omega_up = Msca_up / Mext_up
                p11_up /= Msca_up.reshape(-1, 1)
                g_up /= Msca_up
            else:
                omega_up = omega
                g_up = g
                Mext_up = Mext
                p11_up  = F11

            for k in range(np.size(wvl)):
                phase_function = p11_up[k,:]
                try:
                    gl, alpha = Legendre_coefficients_v1(mu, phase_function, g_up[k], max_moment, x, w, Pl,
                                                         smooth_phase=self.smooth_phase)
                except Exception as e:
                    error_msg = f"wavelength: {wvl[k]}\n"
                    error_msg += f"Error type: {type(e).__name__}\n"
                    error_msg += f"Error message: {e}\n"
                    error_msg += f"  mu: {type(mu)}, shape: {getattr(mu, 'shape', 'N/A')}, dtype: {getattr(mu, 'dtype', 'N/A')}\n"
                    error_msg += f"  phase_function (p11_up[k,:]): {type(p11_up[k, :])}, shape: {getattr(p11_up[k, :], 'shape', 'N/A')}\n"
                    error_msg += f"  max_moment: {type(max_moment)}, value: {max_moment}\n"
                    error_msg += f"  x: {type(x)}, shape: {getattr(x, 'shape', 'N/A')}\n"
                    error_msg += f"  w: {type(w)}, shape: {getattr(w, 'shape', 'N/A')}\n"
                    error_msg += f"  Pl: {type(Pl)}, shape: {getattr(Pl, 'shape', 'N/A')}"
                    print(error_msg)
                file = os.path.normpath(os.path.join(self.output_path,f"layer={i}_wvl={wvl[k]:.3f}um.txt"))
                write_SSP(file, wvl[k], omega_up[k], g_up[k], Mext_up[k],nAngle, 1, idGauss,theta, p11_up[k,:], max_moment, alpha)

    @property
    def num_layer(self):
        """Get the number of layers."""
        return self._num_layer

    @property
    def num_nonice(self):
        """Get the number of non-ice particle types."""
        return self._num_nonice

    @num_layer.setter
    def num_layer(self, value):
        if value == self._num_layer:
            return
        if not isinstance(value, int):
            raise TypeError(f"Number of layers must be an integer. Input value is {value}")
        if value <= 0:
            raise ValueError(f"Number of layers must be positive. Input value is {value}")
        if value > self.ct.max_layer:
            raise ValueError(f"Maximum number of layers is {self.ct.max_layer}. Input value is {value}")
        self._num_layer = value

    @num_nonice.setter
    def num_nonice(self, value):
        if value == self._num_nonice:
            return
        if not isinstance(value, int):
            raise TypeError(f"Number of non-ice particle types must be an integer. Input value is {value}")
        if value < 0:  # Allow zero for no non-ice particles
            raise ValueError(f"Number of non-ice particle types cannot be negative. Input value is {value}")
        # Consider using a specific maximum for non-ice particles if available
        max_nonice = getattr(self.ct, 'max_nonice', self.ct.max_layer)
        if value > max_nonice:
            raise ValueError(f"Maximum number of non-ice particle types is {max_nonice}. Input value is {value}")
        self._num_nonice = value

def validate_interpolation_data(x, y):
    """Validate data for interpolation"""
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length. Got {len(x)} and {len(y)}")

    if len(x) < 2:
        raise ValueError(f"Need at least 2 points for interpolation. Got {len(x)} points")

    if not np.all(np.isfinite(x)) :
        raise ValueError("x contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf)")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing")

def read_lap(file):
    with nc.Dataset(file, 'r') as ds:
        var_names = ['lambda', 'theta', 'omega', 'g', 'F11', 'Mext']
        wvl, theta,omega, g, F11, Mext = [
            ds.variables[var][:] if var in ds.variables else 0.0 for var in var_names
        ]
    nAngle = np.size(theta)
    nPhase = 6
    idGauss = 2
    return wvl, omega, g, Mext, nAngle, nPhase, idGauss, theta, F11
def read_lai(file):
    ct = const()
    with nc.Dataset(file, 'r') as ds:
        var_names = ['ss_alb', 'asm_prm', 'ext_cff_mss','prt_dns']
        omega, g, Mext, particle_density = [
            ds.variables[var][:] if var in ds.variables else 0.0 for var in var_names
        ]
        Mext = Mext * 10  # from m2/kg to cm2/g
        particle_density = particle_density / 1000  # from kg/m3 to g/cm3
        if type(omega) is float:
            var_names = ['omega', 'g', 'Mext']
            omega, g, Mext = [
                ds.variables[var][:] if var in ds.variables else 0.0 for var in var_names
            ]
            particle_density = 1.0

    nAngle = 181
    nPhase = 6
    idGauss = 2

    wvl = ct.solar_spectrum

    resol_fak = 1
    rad = np.pi / 180
    help1 = rad / resol_fak
    help2 = rad * 0.25 / resol_fak
    help3 = rad * (180 - 0.25 / resol_fak)
    deg = 180.0 / np.pi
    theta = np.zeros(nAngle)
    dmu = np.zeros(nAngle)
    for i in range(nAngle):
        d_o1 = 2 * np.sin(i * help1) * np.sin(help1 / 2)
        theta[i] = i * help1 * deg
        if i == 0:
            d_o1 = 2 * np.sin(help2) * np.sin(help2)
            theta[i] = help2 * deg
        if i == nAngle - 1:
            d_o1 = 2 * np.sin(help3) * np.sin(help2)
            theta[i] = help3 * deg
        dmu[i] = d_o1

    F11 = np.zeros((480,nAngle))
    for i in range(480):
        F11[i,:] = Henyey_Greenstein(g[i],np.cos(theta/180.0*np.pi),dmu)

    return wvl, omega, g, Mext, particle_density, nAngle, nPhase, idGauss, theta, F11
def readIceSSP(file,remove_diffraction,truncate_forward,use_HG_phase):
    with nc.Dataset(str(file), 'r') as ds:
        var_names = ['wvl','theta','ss_alb', 'asm_prm', 'ext_cff_mss']
        wvl, theta, omega, g, Mext = [
            ds.variables[var][:] if var in ds.variables else 0.0 for var in var_names
        ]
        if remove_diffraction:
            F11 = ds.variables['F11go'][:]
        else:
            if 'F11' in ds.variables:
                F11 = ds.variables['F11'][:]
            elif 'F11go' in ds.variables:
                F11 = ds.variables['F11go'][:]
            else:
                F11 = 0.0
    Mext = Mext * 10  # from m2/kg to cm2/g
    wvl = wvl * 1E6 # from m to um
    nAngle = 181
    nPhase = 6
    idGauss = 2
    resol_fak = 1
    rad = np.pi / 180
    help1 = rad / resol_fak
    help2 = rad * 0.25 / resol_fak
    help3 = rad * (180 - 0.25 / resol_fak)
    deg = 180.0 / np.pi
    dmu = np.zeros(nAngle)

    if remove_diffraction:
        omega = 2.0 * omega - 1.0
        cal_g = True
        g = np.zeros(np.size(wvl))
    else:
        cal_g = False

    for i in range(nAngle):
        d_o1 = 2 * np.sin(i * help1) * np.sin(help1 / 2)
        theta[i] = i * help1 * deg
        if i == 0:
            d_o1 = 2 * np.sin(help2) * np.sin(help2)
            theta[i] = help2 * deg
        if i == nAngle-1:
            d_o1 = 2 * np.sin(help3) * np.sin(help2)
            theta[i] = help3 * deg
        dmu[i] = d_o1
        help4 = np.cos(theta[i] * rad) * d_o1
        if cal_g:
            g = g + F11[:,i] * help4 * 0.5

    if truncate_forward:
        F11[:,0] = F11[:,1]
    if use_HG_phase:
        F11 = np.zeros((480,nAngle))
        for i in range(480):
            F11[i, :] = Henyey_Greenstein(g[i], np.cos(theta / 180.0 * np.pi), dmu)
    return wvl, omega, g, Mext, nAngle, nPhase, idGauss, theta, dmu, F11
def readSSP(file,remove_diffraction,truncate_forward,use_HG_phase):
    ct = const()
    with nc.Dataset(file, 'r') as ds:
        var_mapping = {
            'omega': 'omega',
            'F11': 'F11',
            'Fgo': 'F11go',
            'Qext': 'Qext',
            'Mext': 'Mext',
            'g': 'g'  # 新增：如果存在变量g，则读取
        }
        data = {}
        for file_var, local_var in var_mapping.items():
            if file_var in ds.variables:
                data[local_var] = ds.variables[file_var][:]
            else:
                data[local_var] = None

        omega = data['omega']
        F11 = data['F11']
        F11go = data['F11go']
        Qext = data['Qext']
        Mext = data['Mext']
        g = data['g']

    if remove_diffraction and (F11go is not None):
        F11 = F11go
        omega = 2.0 * omega - 1.0
        if Qext is not None:
            Mext = Mext * (Qext - 1.0) / Qext
            Qext = Qext - 1.0
    if truncate_forward:
        F11[:,0] = F11[:,1]

    nAngle = 181
    nPhase = 6
    idGauss = 2
    nwvl, _ = F11.shape
    wvl = ct.solar_spectrum[0:nwvl]

    resol_fak = 1
    rad = np.pi / 180
    help1 = rad / resol_fak
    help2 = rad * 0.25 / resol_fak
    help3 = rad * (180 - 0.25 / resol_fak)
    deg = 180.0 / np.pi
    rtddim = 180
    if g is None:
        cal_g = True
        g = np.zeros(np.size(wvl))
    else:
        cal_g = False
    theta = np.zeros(nAngle)
    dmu = np.zeros(nAngle)
    for i in range(nAngle):
        d_o1 = 2 * np.sin(i * help1) * np.sin(help1 / 2)
        theta[i] = i * help1 * deg
        if i == 0:
            d_o1 = 2 * np.sin(help2) * np.sin(help2)
            theta[i] = help2 * deg
        if i == nAngle-1:
            d_o1 = 2 * np.sin(help3) * np.sin(help2)
            theta[i] = help3 * deg
        dmu[i] = d_o1
        help4 = np.cos(theta[i] * rad) * d_o1
        if cal_g:
            g = g + F11[:,i] * help4 * 0.5

    if  use_HG_phase:
        F11 = np.zeros((480,nAngle))
        for i in range(480):
            F11[i, :] = Henyey_Greenstein(g[i], np.cos(theta / 180.0 * np.pi), dmu)

    return wvl, omega, g,  Mext, nAngle, nPhase, idGauss, theta, dmu, F11


def write_SSP(file, wvl, omega, g,  Mext,nAngle, nPhase, idGauss,theta, F11, max_moment, alpha):
    coalbedo = 1.0-omega
    with open(file,'w',encoding='utf-8') as f:
        f.write(f"  {'wvl(um)':<15s}{'coalbedo':<15s}{'<COS>':<15s}{'Mext(cm2/g)':<15s}\n")
        f.write(f"{wvl:15.7E}{coalbedo:15.7E}{g:15.7E}{Mext:15.7E}\n")
        f.write("Number of angles,  number of phase elements, whether to use Gaussian quadrature points \n")
        f.write(f"{nAngle:5}{nPhase:5}{idGauss:5} {max_moment}\n")
        f.write(f"{'Angle':<8s}{'F11':<15s}\n")
        for i in range(nAngle):
            f.write(f"{theta[i]:<6.2f}{F11[i]:15.7E}\n")
        f.write("Maximum Legendre expansion order\n")
        f.write(f"{max_moment}\n")
        f.write("Alpha1\n")
        for i in range(alpha.size):
            f.write(f"{alpha[i]:15.7E}\n")


def fract_mass_to_number(int_or_ext, density, volume_effective_radius, mass_fract):
    v = volume_effective_radius ** 3 * 4 / 3 * np.pi
    Mmix = v[0] * density[0] / (mass_fract[0] + int_or_ext * density[0] * sum(mass_fract[1:] / density[1:]))
    N = np.zeros_like(density)
    N[0] = 1
    N[1:] = Mmix * mass_fract[1:] / (v[1:] * density[1:])
    return Mmix, N
