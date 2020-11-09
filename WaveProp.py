# This code runs wave propagation

from typing import List, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import time
import numba
# from numba import autojit
from numba import jit

import copy

from WavePyClasses import \
    Grid, Model, Kernel, WavePropField, VectorThing, \
    Source, Receiver
from utility_functions import find_nearest, compute_indices
import wavefield_plotting_using_classes as wplot


def calc_absbound(
    grid: Grid, 
    absorbing_boundaries: dict
):

    '''
    initialise Cerjan-type absorbing boundaries (Gaussian taper)
    
    INPUT:
    grid: contains X, Z meshgrids, Lx, Lz (lenght in x & z dirs)
    absorbing_boundaries: a dictionary containing as keys the
        boundaries that are absorbing, and as values the width
        in metres of those boundaries.
    
    OUTPUT:
    absbound: an np array with dimensions of the grid that tapers
        to zero everywhere where there's an absorbing boundary
    '''

    X = grid.X
    Z = grid.Z
    Lx = grid.Lx
    Lz = grid.Lz
    
    absbound = np.ones_like(X)
    
    which_boundaries = absorbing_boundaries.keys()
    
    # left boundary
    if 'left' in which_boundaries:
        width = absorbing_boundaries['left']
        middle = (X>width).astype(int)
        side = (X<=width).astype(int)
        absbound *= middle \
                    + np.exp(-(X-width)**2 / (2*width)**2 ) * side

    # right boundary
    if 'right' in which_boundaries:
        width = absorbing_boundaries['right']
        middle = (X < (Lx-width)).astype(int)
        side = (X >= (Lx-width)).astype(int)
        absbound *= middle \
                    + np.exp(-(X-(Lx-width))**2 / (2*width)**2 ) * side

    # bottom boundary
    if 'top' in which_boundaries:
        width = absorbing_boundaries['top']
        middle = (Z > width).astype(int)
        side = (Z <= width).astype(int)
        absbound *= middle \
                    + np.exp(-(Z-width)**2 / (2*width)**2 ) * side

    # top boundary
    if 'bottom' in which_boundaries:
        width = absorbing_boundaries['bottom']
        middle = (Z < (Lz-width)).astype(int)
        side = (Z >= (Lz-width)).astype(int)
        absbound *= middle \
                    + np.exp(-(Z-(Lz-width))**2 / (2*width)**2 ) * side
                    
    return absbound


@jit(nopython=True)
def div_s_PSV(
    # vector_fields: WavePropField, 
    sxx: np.ndarray,
    sxz: np.ndarray,
    szz: np.ndarray,
    # grid: Grid, 
    dx: float, dz: float, 
    nx: int, nz:int,
    order: int=4):

    '''
    Compute the divergence of the stress field.
    The stress field is a symmetric tensor of shape
     |sxx sxz|
     |szx szz|
    where sxz = szx
    
    INPUT:
    grid: the computational grid, containing at the
          very least nx, nz, dx and dz.
    sxx, sxz, szz: the components of the stress tensor
    order: the computational order (not relevant at the mo)
    
    OUTPUT:
    DS_x, DS_z: the x and z components of the divergence of
                the stress field.
    '''

    DS_x = np.zeros_like(sxx)
    DS_z = np.zeros_like(sxx)

    if order==4:
        # dsxx/dx
        for ii in range(2, nx-2):
            DS_x[:,ii] = 9*(sxx[:,ii]-sxx[:,ii-1]) / (8*dx) - \
                        (sxx[:,ii+1]-sxx[:,ii-2]) / (24*dx)
        # dsxz/dz = dszx/dz
        for jj in range(2, nz-2):
            DS_x[jj,:] = DS_x[jj,:] + \
                         +9*(sxz[jj,:]-sxz[jj-1,:]) / (8*dz) \
                         - (sxz[jj+1,:]-sxz[jj-2,:]) / (24*dz)

        # dszx/dx = dsxz/dx
        for ii in range(2, nx-2):
            DS_z[:,ii] = 9*(sxz[:,ii]-sxz[:,ii-1]) / (8*dx) - \
                        (sxz[:,ii+1]-sxz[:,ii-2]) / (24*dx)
        # dszz/dz
        for jj in range(2, nz-2):
            DS_z[jj,:] = DS_z[jj,:] + \
                         +9*(szz[jj,:]-szz[jj-1,:]) / (8*dz) \
                         - (szz[jj+1,:]-szz[jj-2,:]) / (24*dz)

        return DS_x, DS_z
    else:
        raise ValueError('`order` must be 4. Soz.')


@jit(nopython=True)
def grad_vecfield(
    vx: np.ndarray, vz: np.ndarray,
    dx: float, dz: float, 
    nx: int, nz:int,
    order: int=4):

    '''
    Calculate the gradient of a vector field defined by vx, vz
    effectively used for calculating the gradient of the velocity
    field in P-SV wave propagation
    '''

    dvxdx = np.zeros_like(vx)
    dvxdz = np.zeros_like(vx)
    dvzdx = np.zeros_like(vx)
    dvzdz = np.zeros_like(vx)

    if order == 4:
        # dvx/dx
        for ii in range(1,nx-2):
            dvxdx[:,ii] = 9*(vx[:,ii+1]-vx[:,ii]) / (8*dx) \
                         - (vx[:,ii+2]-vx[:,ii-1]) / (24*dx)

        # dvx/dz
        for jj in range(1,nz-2):
            dvxdz[jj,:] = 9*(vx[jj+1,:]-vx[jj,:]) / (8*dz) \
                         - (vx[jj+2,:]-vx[jj-1,:]) / (24*dz)

        # dvz/dx
        for ii in range(1,nx-2):
            dvzdx[:,ii] = 9*(vz[:,ii+1]-vz[:,ii]) / (8*dx) \
                         - (vz[:,ii+2]-vz[:,ii-1]) / (24*dx)

        # dvx/dz
        for jj in range(1,nz-2):
            dvzdz[jj,:] = 9*(vz[jj+1,:]-vz[jj,:]) / (8*dz) \
                         - (vz[jj+2,:]-vz[jj-1,:]) / (24*dz)

        return dvxdx,dvxdz,dvzdx,dvzdz
    else:
        raise ValueError('Soz, order must be 4.')


def compute_kernels(
    kernels: Kernel, 
    fw_it, fw_fields, 
    vector_fields: WavePropField,
    grid: Grid, 
    sfe: int, 
    dt: float, 
    order: int=4, 
    return_interaction=False):

    '''
    Function to compute sensitivity kernels on-the-fly for
    density, mu and lambda
    '''
    
    # get adjoint wavefields
    vx = vector_fields.vx
    vz = vector_fields.vz
    ux = vector_fields.ux
    uz = vector_fields.uz    

    # grab correct iteration of fwd fields
    vx_fw = fw_fields['vx_fw'][fw_it]
    vz_fw = fw_fields['vz_fw'][fw_it]
    ux_fw = fw_fields['ux_fw'][fw_it]
    uz_fw = fw_fields['uz_fw'][fw_it]

    # compute strain tensor for adjoint fields
    duxdx, duxdz, duzdx, duzdz = grad_vecfield(
        ux, uz, 
        grid.dx, grid.dz, grid.nx, grid.nz,
        order=order)

    # compute strain tensor for forward fields
    duxdx_fw, duxdz_fw, duzdx_fw, duzdz_fw = grad_vecfield(
        ux_fw, uz_fw, 
        grid.dx, grid.dz, grid.nx, grid.nz,
        order=order)

    # compute interaction between fw & adj wavefields and then
    # add (subtract) the interation to (from) the already existing knls:

    ## rho
    int_rho = vx * vx_fw + vz * vz_fw
    kernels.rho -= int_rho * sfe * dt

    ## mu
    int_mu = 2.*duxdx*duxdx_fw + 2.*duzdz*duzdz_fw  \
             + (duxdz + duzdx) * (duxdz_fw + duzdx_fw)
    kernels.mu -= int_mu * sfe * dt

    ## lambda
    int_lambda = (duxdx + duzdz) * (duxdx_fw + duzdz_fw)
    kernels.lambd -= int_lambda * sfe * dt

    if return_interaction:
        return kernels, int_rho, int_mu, int_lambda
    else:
        return kernels


def run_waveprop(
    sources: Union[List[Source], Source], 
    receivers: Union[List[Receiver], Receiver], 
    model: Model,
    absorbing_boundaries: dict = {}, 
    computational_order: int=4,
    simulation_mode: str='forward',
    store_forward_fields=False, 
    store_forward_every=10,
    forward_fields=None,
    plot_wavefield=False, 
    plot_wavefield_every=10,
    return_last_wavefield = None,
    verbose=False, veryverbose=False
):

    '''
    Run wave propagation either in forward or adjoint mode.
    Forward means: 
        A seismic source emits a signal, the resultin waves
        propagate through the model and if any receivers are
        present, these will record any vibrations reaching them.
    Adjoint means:
        A receiver has an adjoint source (based on a forward
        simulation), from which signal propagates through the
        model (technically in reverse time). As this simulation
        takes place, the interaction with this adjoint field with
        a previously computed forward field is computed. This 
        results in sensitivity kernels.
        It is assumed that a forward simulation has already been 
        carried out in which forward fields were stored.
        It is also assumed that adjoint sources have been created.
    
    # Forward - not storing fields
    receivers = run_waveprop(
        sources, receivers, model, absorbing_boundaries,
        simulation_mode='forward',
        store_forward_fields=False, ...)
    
    # Forward -- storing fields
    receivers, fw_fields = run_waveprop(
        ..., store_forward_fields = True, ...)
    
    # Adjoint - using previously stored fields
    kernels = run_waveprop(
        ..., simulation_mode='adjoint',
             forward_fields=fw_fields, ...)

    '''

    # == Preliminary input checks ==

    allowed_simulation_modes = ['forward', 'adjoint']
    if simulation_mode not in allowed_simulation_modes:
        raise ValueError('simulation mode must be one of '+
              ', '.join(allowed_simulation_modes)
              )

    if verbose:
        print('Simulation mode: {}'.format(simulation_mode))

    # check if necessary input is present:
    if simulation_mode == 'adjoint':
        if not forward_fields:
            raise ValueError('Adjoint mode: input forward fields are necessary.')

        # if 'adstf' not in receivers[0].keys():
        if not isinstance(receivers[0].adstf, VectorThing):
            raise ValueError(
                'Adjoint mode: first append adjoint ',
                'source time functions to the receivers.'
            )

    if return_last_wavefield:
        allowed_wavefields = ['vx','vz']
        if return_last_wavefield not in allowed_wavefields:
            raise ValueError(
                'return_last_wavefield must be one of'+
                ', '.join(allowed_wavefields)
            )
            
    if isinstance(sources, Source):
        sources = [sources]
    if isinstance(receivers, Receiver):
        receivers = [receivers]

    # == Initialisation of values / fields ==

    # initialise some values
    grid = model.grid
    dx = grid.dx
    dz = grid.dz
    nt = sources[0].time.nt
    dt = sources[0].time.dt
    tmax = sources[0].time.t_max
    sfe = store_forward_every

    # initialise source  & receiver indices. Ugly.
    for src in sources:
        src.loc_idx = compute_indices(
            src.location.x, src.location.z,
            grid.X, grid.Z
            )
    for rec in receivers:
        rec.loc_idx = compute_indices(
            rec.location.x, rec.location.z,
            grid.X, grid.Z
            )

    # initialise absorbing boundaries
    absbound = calc_absbound(
        grid, 
        absorbing_boundaries)

    # check some shizzle
    if verbose:
        print('Lx, Lz   = {:.1f}, {:.1f}'.format(Lx, Lz))
        print('dx, dz    = {:.1f}, {:.1f} m'.format(
            grid.dx, grid.dz))
        print('nx, nz    = {:d}, {:d}'.format(
            grid.nx, grid.nz))
        print('timestep  = {:.2g} s'.format(dt))
        print('duration  = {:.1f} s'.format(tmax))
        print('timesteps = {:d} (running from {:d} to {:d})'.format(nt, 0, nt-1))
        
        if simulation_mode == 'forward' and store_forward_fields:
            print(
                'storing wavefield every {:d} timesteps '.format(sfe),
                '(t = {:.2f}, {:.2f}, ... {:.2f} s)'.format(0., dt*sfe, dt*(nt-1))
            )
        elif simulation_mode == 'adjoint':
            print(
                'retrieving wavefield every {:d} timesteps '.format(sfe),
                '(t = {:.2f}, {:.2f}, ... {:.2f} s)'.format(dt*(nt-1), dt*(nt-1-sfe), 0.)
            )

    # initialise stress fields etc.
    vector_fields = WavePropField(grid)

    # initialise to-be-stored forward fields
    if store_forward_fields:
        fw_fields = {'ux_fw': [], 'uz_fw': [],
                     'vx_fw': [], 'vz_fw': [],
                     'vector': []}

    # adjoint mode: initialise kernels
    if simulation_mode == 'adjoint':
        knls = Kernel(grid)

    # initialise wavefield plotting
    if plot_wavefield:

        print('plotting the wavefield')
        plt.ion()

        if simulation_mode == 'forward':
            stf = sources[0].stf
            title = 'X velocity field, t = {:.2f}'.format(0.)
            prefac = 1e-12
        elif simulation_mode == 'adjoint':
            stf = receivers[0].adstf
            title = 'X adjoint velocity field, t = {:.2f}'.format(nt*dt)
            prefac = 1e-7
        src_amp = [np.linalg.norm((x,z)) for x,z in zip(stf.x, stf.z)]
        cmaks = prefac*np.max(src_amp)

        fig, ax = plt.subplots(1, figsize=(10,5))
        _, ax, fig = wplot.plot_field(
            grid, vector_fields.vx, sources, receivers,
            title=title,
            cmaks=cmaks, ax=ax,draw=True)

    # initialise seismograms
    if simulation_mode == 'forward':
        for rec in receivers:
            rec.seismogram.x = []
            rec.seismogram.z = []

    # The wave propagation loop
    for it in range(nt):

        # store forward _displacement_ field
        if (simulation_mode == 'forward' 
            and store_forward_fields 
            and it % sfe == 0
        ):
            fw_fields['ux_fw'].append(vector_fields.ux.copy())
            fw_fields['uz_fw'].append(vector_fields.uz.copy())

        # compute divergence of current stress
        DSX, DSZ = div_s_PSV(
            vector_fields.sxx, vector_fields.sxz, vector_fields.szz,
            grid.dx, grid.dz, grid.nx, grid.nz,
            computational_order)

        # add point sources
        if simulation_mode == 'forward':
            for src in sources:
                DSX[src.loc_idx] += src.stf.x[it] /dx/dz
                DSZ[src.loc_idx] += src.stf.z[it] /dx/dz
        elif simulation_mode == 'adjoint':
            for rec in receivers:
                DSX[rec.loc_idx] += rec.adstf.x[it]
                DSZ[rec.loc_idx] += rec.adstf.z[it]

        # update velocity field
        vector_fields.vx += dt*DSX/model.rho
        vector_fields.vz += dt*DSZ/model.rho

        # apply absbound
        vector_fields.vx *= absbound
        vector_fields.vz *= absbound

        # compute gradient of velocity field
        dvxdx,dvxdz,dvzdx,dvzdz = grad_vecfield(
            vector_fields.vx, vector_fields.vz, 
            grid.dx, grid.dz, grid.nx, grid.nz,
            computational_order)

        # update stress tensor
        vector_fields.sxx += \
            dt*( (model.lambd + 2*model.mu) * dvxdx[:,:] \
            + model.lambd * dvzdz[:,:] )
        vector_fields.szz += \
            dt*( (model.lambd + 2*model.mu) * dvzdz[:,:] \
            + model.lambd * dvxdx [:,:] )
        vector_fields.sxz += \
            dt*( model.mu * (dvxdz[:,:] + dvzdx[:,:]) )

        # compute displacement field
        vector_fields.ux += vector_fields.vx*dt
        vector_fields.uz += vector_fields.vz*dt

        # store seismograms
        if simulation_mode == 'forward':
            for rec in receivers:
                rec.seismogram.x.append( vector_fields.vx[rec.loc_idx] )
                rec.seismogram.z.append( vector_fields.vz[rec.loc_idx] )


        # store forward _velocity_ field (and iter vector)
        if (simulation_mode == 'forward' 
            and store_forward_fields 
            and it % sfe == 0
        ):
            fw_fields['vector'].append(it)
            fw_fields['vx_fw'].append(vector_fields.vx.copy())
            fw_fields['vz_fw'].append(vector_fields.vz.copy())

        # convenience definition of timestep in opposite direction
        it_fw = nt-1-it

        # if adjoint mode: compute current kernel contribution
        # & add this to the real kernel (to be given back to )
        if (simulation_mode == 'adjoint' 
            and it_fw % sfe == 0
        ):

            it_field = int(it_fw / 10) # why 10?! should be sfe?
            
            if veryverbose:
                print('it: {:.1f} (fwd: {:.1f})'.format(it, it_fw))
                print('forward field it: {:.2f}'.format(it_field))
            
            knls = compute_kernels(
                knls, it_field,
                forward_fields, 
                vector_fields,
                grid, sfe, dt, computational_order,
                # return_interaction=True
            )

        # plot wavefield
        if plot_wavefield:

            if it % plot_wavefield_every == 0:
                
                # determine which wf is plotted
                if isinstance(plot_wavefield, bool):
                    plot_wavefield = 'vx'
                if plot_wavefield == 'vx':
                    field = vector_fields.vx
                elif plot_wavefield == 'vz':
                    field = vector_fields.vz

                # make title
                if simulation_mode == 'forward':
                    title = '{} velocity field, t = {:.2f}'.format(plot_wavefield[1].upper(), it*dt)
                elif simulation_mode == 'adjoint':
                    title = '{} adjoint velocity field, t = {:.2f}'.format(plot_wavefield[1].upper(), it_fw*dt)

                ax.cla()
                wplot.plot_field(
                    grid, field, sources, receivers,
                    title=title,
                    cmaks=cmaks, ax=ax,draw=True, updating=False)

                # img.set_array(vx.ravel())

                fig.canvas.draw()
                # fig.canvas.flush_events()
                time.sleep(0.01)

        else:
            progress_report_time = int(nt / 4)
            if it % progress_report_time == 0:
                print('executed {:.0f}% of wave propagation'.format(it/nt*100))
            elif it == nt-1:
                print('done')
                
    if verbose:
        print('final timestep is {} (number {})'.format(it*dt, it))

    # add time axis to receivers for convenience
    for rec in receivers:
        rec.time = copy.deepcopy(sources[0].time)

    if return_last_wavefield == 'vx':
        last_wavefield = vector_fields.vx
    elif return_last_wavefield == 'vz':
        last_wavefield = vector_fields.vz

    # return shizzle
    if simulation_mode=='forward' and store_forward_fields:
        return receivers, fw_fields
    elif simulation_mode == 'adjoint':
        if return_last_wavefield:
            return knls, last_wavefield
        return knls
    else:
        if return_last_wavefield:
            return receivers, last_wavefield
        return receivers

# Functionality below is a bit ugly, all...
def make_adjoint_source(
    receivers: List[Receiver], 
    pick, 
    misfit='cc_time_shift', 
    plot=3):

    allowed_misfits = ['cc_time_shift', 'L2norm']
    if misfit not in allowed_misfits:
        raise ValueError(
            'misfit must be one of: ', ', '.join(allowed_misfits))

    def get_taper(t, t_min, t_max, taper_width):

        tw = np.ones(np.shape(t)) * (t>t_min) * (t<t_max)
        tw += (0.5 + 0.5*np.cos(np.pi*(t_max-t)/(taper_width))) \
                * (t>=t_max) * (t<t_max+taper_width)
        tw += (0.5+0.5*np.cos(np.pi*(t_min-t)/(taper_width))) \
            * (t>t_min-taper_width)*(t<=t_min)

        return tw

    # windowing a seismogram
    def window_seismogram(seis, t, pick):
        # [idx0, idx1] = [int(x / dt) for x in pick['times']]
        # seisw = np.zeros(np.shape(seis))
        # seisw[idx0:idx1] = [x for x in seis[idx0:idx1]]
        taper_width = t[-1] / 20
        tw = get_taper(t, *pick['times'], taper_width)
        seisw = seis * tw
        return seisw

    # misfit fnal: cross-correlation time shift (dataless)
    def cc_time_shift(t, v):
        dt = t[1] - t[0]
        adstf = v / np.inner(v,v) / dt
        adstf = adstf[::-1] * -1. # -1 factor for velocity seismograms
        return np.asarray(adstf)

    # misfit fnal: L2 norm (dataless)
    def L2norm(t, v):
        adstf = copy.deepcopy(v)
        # adstf = list(reversed(adstf))
        adstf = adstf[::-1]
        # adstf = [-x for x in adstf] # for velocity seismograms
        return adstf

    # create adstf
    receivers_copy = copy.deepcopy(receivers)
    # define time axis, dt
    t = receivers[0].time.t
    dt = receivers[0].time.dt

    if plot>0:
        _, axes = plt.subplots(plot, 1, figsize=(10,2+plot*2))
        a0 = axes if plot == 1 else axes[0]    

    for rec in receivers_copy:

        # plot velocity seismogram
        if plot:
            rec.plot_seismogram(
                ax=a0, show_grid=True, show=False)

        for comp, seis_v in [('x', rec.seismogram.x), 
                           ('z', rec.seismogram.z)]:

            if misfit == 'L2norm':
                seis = np.cumsum(seis_v)*dt
            else:
                seis = seis_v

            adstf = np.zeros(np.shape(seis))
            seis_windowed = np.zeros(np.shape(seis))

            # only compute if we're at the right pick component
            if comp not in pick['component']:
                rec.set_adstf(adstf, comp)
                continue
            
            if pick['times'][0] > rec.time.t_max:
                raise ValueError(
                    'Your pick starts later than the end of '
                    'the simulation. Please repick.'
                )

            seis_windowed = window_seismogram(seis, t, pick)

            if misfit=='cc_time_shift':
                adstf = cc_time_shift(rec.time.t, seis_windowed)
            elif misfit=='L2norm':
                adstf = L2norm(rec.time.t, seis_windowed)

            rec.set_adstf(adstf, comp)

            if plot>1:
                axes[1].plot(rec.time.t, seis_windowed,
                        label='windowed seismogram ({} component)'.format(comp))
                axes[1].legend()
                axes[1].grid('on')
                axes[1].set_xlabel('time [s]')
                axes[1].set_ylabel('amplitude [m/s]')
            if plot>2:
                axes[2].plot(rec.time.t, adstf,
                        label='adjoint source, time reversed ({} component)'.format(comp))
                axes[2].legend()
                axes[2].grid('on')
                axes[2].set_xlabel('adjoint (reverse) time [s]')
                axes[2].set_ylabel('adjoint amplitude')

    if plot>0:
        plt.show()

    return receivers_copy
