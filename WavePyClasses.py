import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Union, List

from FresnelZone import Fresnel_width_from_velocity_field_and_stf as Fresnel_width
from FresnelZone import plot_Fresnel

# import wavefield_plotting_using_classes as wplot

class VectorThing:
    
    '''
    Class that, quite generically, has an x and a z
    which are ndarrays or floats.
    '''
    
    def __init__(
        self,
        z: Union[np.ndarray, float]=None,
        x: Union[np.ndarray, float]=None,
        ):
                
        self.x = x
        self.z = z
        
    def asarray(self):
        
        '''
        Return a numpy asarray of [self.x, self.z]
        '''
        
        return np.asarray([self.x, self.z])


class TimeAxis:
    
    '''
    Class that describes a time axis with some handy
    extra attributes (e.g. timestep dt, t_max, number of 
    timesteps)
    '''
    
    def __init__(self, t_max, dt, verbose=False):
        
        '''
        Initialise a time axis
        '''
        
        # define time axis
        nt = int(t_max / dt)
        t_max = nt*dt
        self.t = np.linspace(0, t_max, nt+1) # (start, stop, step)
        
        # set other time parameters
        self.dt = dt
        self.t_max = t_max
        self.nt = len(self.t)

        if verbose:
            print(
                "t_max set to {} such that it is a whole multiple of dt".format(self.t_max)
            )
        

class PointTimeSeries:
    
    '''
    A class defining the shape of a time series for a point.
    This could be a source (with time series stf), or a receiver
    (with time series seismogram).
    All these time series have a direction (i.e. an x and a z 
    component) and have a time axis.
    '''
    
    def __init__(
        self, 
        loc_x: Optional[float]=None,
        loc_z: Optional[float]=None,
        t_max: Optional[float]=None, 
        dt: Optional[float]=None,
    ):

        '''
        Initialise (empty) time axis, location and location index
        '''
        
        # time params
        if t_max and dt:
            self.time=TimeAxis(t_max, dt)
        else:
            self.time = None
        
        # location - leave empty or set properly
        self.location = VectorThing(x=loc_x, z=loc_z)
        
        # preliminary location index, as list [x,z] -- empty
        self.loc_idx = []
        
    def set_time(self, t_max:float, dt:float):
        
        '''Set a time axis'''
        
        self.time = TimeAxis(t_max, dt)
    
    def set_location(self, loc_x:float, loc_z:float):
        
        '''Set a location for the point'''
        
        self.location = VectorThing(x=loc_x, z=loc_z)


class Source(PointTimeSeries):

    '''
    A class for sources, derived from the general PointTimeSeries.
    
    Depending on the input, you can immediately set:
    - location (requires loc_x, loc_z)
    - time axis (requries t_max, dt)
    - the source time function stf.
      (defining this requires src_type, src_direction,
      plus a number of kwargs depending on the src type)
      
    In order to set a stf, the time axis must be defined.
      
    Currently the only implemented src_type is 'ricker'.
    '''
    
    def __init__(
        self,
        loc_x: Optional[float]=None,
        loc_z: Optional[float]=None,
        t_max: Optional[float]=None, 
        dt: Optional[float]=None,
        # source types
        src_type: Optional[str]=None, 
        src_direction: Optional[list]=None,
        # t_0: Optional[float]=None, # for ricker
        # tau: Optional[float]=None, # for ricker
        verbose: bool=False,
        **kwargs # for all the stf-specific arguments
    ):
        '''
        Set the initial source attributes. This can include:
        - location
        - time axis
        - source time function
        '''
        
        # inherit everything from super
        super().__init__(loc_x, loc_z, t_max, dt)
        
        # initialise the stf (empty for now)
        self.stf = VectorThing()

        # set stf, if enough info supplied
        if src_type and not src_direction:
            raise ValueError(
                'To define a source-time function, you need '
                'to define a source direction as well.'
            )
        if src_type and src_direction:
            self.set_stf(src_type, src_direction, **kwargs)


    def set_stf(
        self, 
        src_type: str, 
        src_direction: list, 
        **kwargs):
        
        '''Set the source-time function'''

        if not self.time:
            raise ValueError(
                'You must first define a time axis using '
                'Source.set_time()'
                )
        
        # currently only "ricker" is implemented
        if src_type == 'ricker':
            if ('t_0' not in kwargs.keys() or 
                'tau' not in kwargs.keys()
            ):
                raise ValueError(
                    "for source type 'ricker', need t_0 and tau."
                    )
                
            stf_directionless = self.ricker(**kwargs)
        else:
            raise ValueError("Only available stf_type is 'ricker'")
        
        # store directionless stf
        stf_directionless = np.asarray(stf_directionless)
        self.stf_directionless = stf_directionless
        
        # determine the source norm
        source_norm = np.linalg.norm(src_direction)
        self.src_direction = np.asarray(src_direction) / source_norm
        # src_x = src_direction[0] / source_norm
        # src_z = src_direction[1] / source_norm
        
        # set stf itself
        self.stf.x = self.src_direction[0] * stf_directionless
        self.stf.z = self.src_direction[1] * stf_directionless

    def ricker(self, t_0: float, tau: float):

        '''
        Give back a ricker wavelet based on the time axis of this
        source,
        duration and zero crossing
        t     = time axis
        t_0   = location of middle of ricker wavelet
        tau   = source duration
        '''
        
        tau_0=2.628 # predefined variable to normalise the source duration

        alpha = 2. * tau_0 / tau
        stf = (-2.*alpha**3/np.pi) * (self.time.t-t_0) * \
            np.exp(-alpha**2 * (self.time.t-t_0)**2 )
            
        return np.asarray(stf)
    
    
    def __str__(self):  
        s = "Source properties:\n"
        if self.location.x:
            s += " location (x,z): {:.2f}, {:.2f} m \n".format(self.location.x, self.location.z)
        if self.time:
            s += " time: from {:.2f} to {:.2f} s (dt={:.2f} s, {} timesteps)\n".format(self.time.t[0], self.time.t_max, self.time.dt, self.time.nt)
        if self.loc_idx:
            s += " location index (x,z): "+str(self.loc_idx)+"\n"
        if self.stf.x is not None:
            s += " source-time-function has been defined\n"
            
        return s


    def plot_stf(
        self,
        plot_spectrum: bool=True,
        return_fig: bool=False
        ):

        dt = self.time.dt
        t = self.time.t
        try:
            stf = self.stf_directionless
        except AttributeError as e:
            print('source-time function has not been defined.')
            raise e
        
        # prepare figure
        if plot_spectrum:
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,5))
        else:
            fig, ax1  = plt.subplots(1,1, figsize=(10,2.5))

        # first subplot (always present): source time function in time domain

        ax1.plot(t, stf)
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('source time function amplitude')
        
        # second subplot (optional): amplitude spectrum in frequency domain

        if plot_spectrum:
            
            # first compute the frequency domain spectrum
            spec = np.fft.fft(stf) # fast fourier transform
            freq = np.fft.fftfreq(spec.size, d = dt / 4.) # frequency axis

            # then plot frequency and spectrum
            ax2.plot(np.abs(freq), np.abs(spec))
            ax2.set_xlabel('frequency [Hz]')
            ax2.set_ylabel('amplitude spectrum')
            # hardcoded and ugly but functional:
            ax2.set_xlim([0,10]) # limit the spectrum to 10 Hz for legibility

        plt.show() # actually show the plot
        
        if return_fig:
            return fig


class Receiver(PointTimeSeries):
    
    '''
    A class for receivers, derived from the general PointTimeSeries.

    Depending on the input, you can immediately set:
    - location (requires loc_x, loc_z)
    - time axis (requries t_max, dt)
    '''
    
    def __init__(
        self,
        loc_x: Optional[float]=None,
        loc_z: Optional[float]=None,
        t_max: Optional[float]=None, 
        dt: Optional[float]=None,
    ):
        
        # inherit everything from super
        super().__init__(loc_x, loc_z, t_max=None, dt=None)
        
        # initialise the seismogram
        self.seismogram = VectorThing()
        
        # initialise the adjoint stf
        self.adstf = None # ultimately will be VectorThing

    def __str__(self):  
        s = "Receiver properties:\n"
        if self.location.x:
            s += " location (x,z): {:.2f}, {:.2f} m \n".format(self.location.x, self.location.z)
        if self.time:
            s += " time: from {:.2f} to {:.2f} s (dt={:.2f} s, {} timesteps)\n".format(self.time.t[0], self.time.t_max, self.time.dt, self.time.nt)
        if self.loc_idx:
            s += " location index (x,z): "+str(self.loc_idx)+"\n"
        if self.seismogram.x is not None:
            s += " seismogram has been recorded\n"
        if self.adstf is not None:
            s += " adjoint source has been determined\n"
            
        return s

    def plot_seismogram(
        self, 
        ax=None,
        show_grid=True,
        show_legend=True,
        show=True,
        ):
        
        '''
        Plot the seismogram for this receiver. Both components
        will show up on one axis.
        '''
        
        # preliminary check if all requirements defined
        if not self.time or not self.seismogram.x:            
            missing = []
            if not self.time:
                missing += ['time']
            if not self.seismogram.x:
                missing += ['seismogram']
            raise ValueError(
                "This receiver is missing"
                ", ".join(missing)
            )

        t = self.time.t
        seis = self.seismogram

        if not ax:
            _, ax = plt.subplots(1,1, figsize=(10,4))

        ax.plot(
            t, seis.x, 
            label='velocity seismogram (x component)')
        ax.plot(
            t, seis.z, 
            label='velocity seismogram (z component)')

        ax.set_xlabel('time [s]')
        ax.set_ylabel('velocity [m/s]')

        if show_grid:
            ax.grid('on')
        if show_legend:
            ax.legend()
            
        if show:
            plt.show()
            
    def set_adstf(
        self,
        adstf,
        comp
    ):
        
        '''
        Set adjoint source-time function.
        '''
    
        if not isinstance(self.adstf, VectorThing):
            self.adstf = VectorThing()
        
        if comp == 'x':
            self.adstf.x = adstf
        elif comp == 'z':
            self.adstf.z = adstf


class Grid:
    
    '''
    The grid class, creating a X and Z meshgrid (plus some other stuff)
    from input lengths in x and z directions (Lx, Lz), grid spacing (dx, dz)
    '''

    def __init__(self, Lx: float, Lz: float, dx: float, dz: float):

        nx = int(Lx / dx) +1 # this may slightly alter the ultimate dx
        nz = int(Lz / dz) +1 # this may slightly alter the ultimate dz

        x = np.linspace(0, Lx, nx)
        z = np.linspace(0, Lz, nz)
        self.X, self.Z = np.meshgrid(x, z)

        self.Lx = x[-1]
        self.Lz = z[-1]
        self.nx = len(x)
        self.nz = len(z)
        self.dx = x[1] - x[0]
        self.dz = z[1] - z[0]

    def print_params(self):

        '''Print grid parameters'''

        print('Grid properties:')
        print(' Lx, Lz = {:.2f}, {:.2f} m'.format(self.Lx, self.Lz))
        print(' nx, nz = {}, {}'.format(self.nx, self.nz))
        print(' dx, dz = {:.2f}, {:.2f} m'.format(self.dx, self.dz))


class Model:

    '''
    The model class, creating a 2-D regular grid model with attached params for an elastic, isotropic medium.
    Parameters are density (rho), mu and lambda. 
    '''

    def __init__(self, grid: Grid):

        '''Initialise a wave propagation model'''

        self.grid = grid
        
        self.rho   = np.zeros_like(grid.X)
        self.mu    = np.zeros_like(grid.X)
        self.lambd = np.zeros_like(grid.X)

    @staticmethod
    def reparametrise_rvv_rml(rho, vs, vp):

        '''
        Reparametrise from rho-vs-vp to rho-mu-lambda.
        Returns only mu and lambda
        '''
        
        mu = vs**2 * rho
        la = (vp**2 - 2*vs**2) * rho

        return mu, la
    
    @staticmethod
    def reparametrise_rml_rvv(rho, mu, lambd):
        
        '''
        Reparametrise from rho-mu-lambda to rho-vs-vp.
        Returns only vs and vp
        '''
        
        vs = np.sqrt(mu/rho)
        vp = np.sqrt( (lambd + 2*mu) / rho)
        
        return vs, vp


    def plot_model(self, parametrisation='rhovsvp', show=True):
        
        _, axes = plt.subplots(3, 1, figsize=(10,10))
        
        if parametrisation == 'rhomulambda':
            params = ['rho', 'mu', 'lambda']
            fields = [self.rho, self.mu, self.lambd]
        elif parametrisation == 'rhovsvp':
            params = ['rho', 'vs', 'vp']
            vs, vp = self.reparametrise_rml_rvv(
                self.rho, self.mu, self.lambd)
            fields = [self.rho, vs, vp]
        else:
            raise ValueError(
                'parametrisation not recognised')    
        
        for param, field, ax in zip(params, fields, axes):
            plot_field(
                self.grid,
                field,
                title=param,
                colorbar=True,
                ax=ax,
                )

        if show:
            plt.tight_layout()
            plt.show()


class HomogeneousModel(Model):

    '''
    Homogeneous model with single values for three parameters.
    Derived from Model.
    Class can be initialised with either rho-mu-lambda or rho-vs-vp.
    '''

    def __init__(
        self, grid: Grid,
        rho=None, mu=None, lambd=None, vs=None, vp=None
    ):

        '''
        Initialise a homogeneous model from input values, 
        using either
        - rho-vs-vp
        - rho-mu-lambda
        '''
        
        # inherit everything from super
        super().__init__(grid)

        if rho and vs and vp:
            if mu or lambd:
                print(
                    'Warning: more than 3 params defined. '
                    'Using rho-vs-vp.'
                )
            mu, lambd = self.reparametrise_rvv_rml(rho, vs, vp)
        elif rho and mu and lambd:
            pass
        else:
            raise ValueError(
                'Parametrisation must be one of'+
                'rho-mu-lambda or rho-vs-vp'
            )

        self.rho   = rho*np.ones_like(self.rho)
        self.mu    = mu*np.ones_like(self.mu)
        self.lambd = lambd*np.ones_like(self.lambd)


class Kernel(Model):
    
    '''
    A class for sensitivity kernes, derived from Model.
    Its main difference is that you can reparametrise a kernel
    (kind of a derivative) which requires a model as well,
    and that you can plot the kernels in different 
    parametrisations - both absolute and relative (to a model).
    '''
    
    def __init__(self, grid: Grid):
        
        '''Initialise. Not different from Model.'''
        
        # inherit everything from super
        super().__init__(grid)


    def reparametrise_kernels_rml_rvv(
        self,
        model: Model
    ):
        '''
        Reparametrise the kernel. Requires a Model.
        '''
        
        rho = model.rho
        vs, vp = model.reparametrise_rml_rvv(
            model.rho, model.mu, model.lambd)
        
        k_rho2 = (self.rho + vs**2 * self.mu 
                  + (vp**2 - 2*vs**2) * self.lambd
                  )
        k_vs = (2 * rho * vs * self.mu 
                - rho * vs * self.lambd
                )
        k_vp = 2 * rho * vp * self.lambd
        
        return k_rho2, k_vs, k_vp
    
    def plot_kernels(
        self,
        source: Optional[Source]=None, 
        receiver: Optional[Receiver]=None,
        model: Optional[Model]=None, 
        parametrisation='rhovsvp', mode='relative',
        cmaks = None,
        colour_percentile=99.97,
        plot_Fresnel_zone = False,
        verbose=False,
        show=True
    ):

        '''
        Plot the sensitivity kernels (using plot_field).
        Probably will require a model as well. For extra fancy
        plotting you'll need to supply a source and rec. In that
        case, you can plot the Fresnel zone as well.
        '''

        if parametrisation == 'rhovsvp':

            if not model:
                raise ValueError(
                    'In order to reparametrise, you need to '
                    'supply a model.'
                )
                
            k_rho2, k_vs, k_vp = self.reparametrise_kernels_rml_rvv(model=model)

        if mode == 'relative':
            
            if not model:
                raise ValueError(
                    'For relative kernels, you need to supply a model.'
                )

            vs, vp = model.reparametrise_rml_rvv(
                model.rho, model.mu, model.lambd)
            
            k_rho2 *= model.rho
            k_vs *= vs
            k_vp *= vp
            
        if plot_Fresnel_zone:
            if not source and receiver:
                raise ValueError(
                    'In order to plot a Fresnel zone you need '
                    'to supply a source and a receiver'
                )

        _, axes = plt.subplots(3,1, figsize = (10,10))
            
        if parametrisation == 'rhomulambda':
                params = ['rho', 'mu', 'lambda']
                fields = [self.rho, self.mu, self.lambd]
        elif parametrisation == 'rhovsvp':
            params = ['rho2', 'vs', 'vp']
            fields = [k_rho2, k_vs, k_vp]
        else:
            raise ValueError(
                'parametrisation not recognised')
            
        # determine colour maximum (important for knls)
        if not cmaks:
            allvalues = np.concatenate(fields, axis=None)
            cmaks = np.percentile(allvalues, colour_percentile)
        
        if plot_Fresnel_zone:
            loc_s = source.location.asarray()
            loc_r = receiver.location.asarray()
            w_f = Fresnel_width(
                velocity_field = vp,  # hardcoded
                loc_s=loc_s, loc_r=loc_r, 
                time_axis=source.time.t, 
                stf=source.stf_directionless,
                verbose=verbose
                )
        
        for param, field, ax in zip(params, fields, axes):
            plot_field(
                self.grid,
                field,
                title=param+" sensitivity kernel",
                cmaks=cmaks,
                colorbar=True,
                cmap='seismic',
                ax=ax,
                draw=False,
                )
            
            if plot_Fresnel_zone:
                plot_Fresnel(
                    ax, loc_s, loc_r, w_f, 
                    label="P-wave Fresnel zone",
                    units='km',
                    zorder=1000000000, color='k', linewidth=2, linestyle='--')
                ax.legend()

        plt.subplots_adjust(hspace=0.35)

        if show:
            plt.show()


class WavePropField:
    
    '''
    A convenience class holding the wave propagation fields.
    Not really used other than to shorten things.
    '''
    
    def __init__(self, grid: Grid):
        
        '''
        Initialise all the wave propagation fields
        '''
        
        X = grid.X
        
        self.ux = np.zeros(np.shape(X))
        self.uz = np.zeros(np.shape(X))
        self.vx = np.zeros(np.shape(X))
        self.vz = np.zeros(np.shape(X))
        self.sxx = np.zeros(np.shape(X))
        self.sxz = np.zeros(np.shape(X))
        self.szz = np.zeros(np.shape(X))


# this really shouldn't be here, but I can't solve it otherwise

def plot_field(
    grid: Grid, 
    field, 
    sources: Union[Source, List[Source], None] = None, 
    receivers: Union[Receiver, List[Receiver], None] = None,
    title=None, colorbar=False, 
    cmaks=None, 
    ax=None, draw=True, updating=False,
    **kwargs
):
    
    '''
    Plot the velocity field in a somewhat nice way
    '''
    
    if isinstance(sources, Source):
        sources = [sources]
    if isinstance(receivers, Receiver):
        receivers = [receivers]
    
    if not ax:
        fig, ax = plt.subplots(1, figsize=(10,5))
    else:
        plt.cla()

    if not title:
        print('no plot title given')

    # plot any field
    X = grid.X / 1000.
    Z = grid.Z / 1000.

    if not ax:
        fig, ax = plt.subplots(1, figsize=(10,5))

    if cmaks:
        img = ax.pcolormesh(X,Z, field, shading='gouraud',
                            vmin=-1*cmaks, vmax=cmaks, **kwargs)
    else:
        img = ax.pcolormesh(X,Z, field, shading='gouraud', **kwargs)

    if sources:
        for src in sources:
            loc = src.location.asarray() /1000.
            ax.plot(*loc, c= 'k', marker='*')
    if receivers:
        for rec in receivers:
            loc = rec.location.asarray() /1000.
            ax.plot(*loc, c='k', marker='^')

    ax.set_xlabel('horizontal distance [km]')
    ax.set_ylabel('depth [km]')
    if title:
        ax.set_title(title)
    
    # The below is a really ugly hack to make sure that 
    # the axis isn't continuously flipping up & down when you're
    # plotting a simulation as it is progressing. The annoying 
    # thing is that it depends entirely on the version of python, 
    # matplotlib, and whatnot. I don't want to think about it. 
    # Whatever.
    if not updating:  
        ax.invert_yaxis()

    ax.axis('image')

    if colorbar:
      plt.colorbar(mappable=img, ax=ax)

    fig = plt.gcf()
    if draw:
        fig.canvas.draw()

    return img, ax, fig
