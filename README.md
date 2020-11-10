# WavePy
Basic seismic wave propagation code for teaching purposes (python based). With this code, you can:

* run elastic seismic wavefield simulations
* select data windows
* compute sensitivity kernels

A quick demo is given in the `Quick_demo_WavePy.ipynb` Jupyter notebook, while there is also a more elaborate wave propagation practical notebook `Wave_propagation_practical.ipynb`. 

You can launch this environment (and run the notebooks interactively) in a mybinder live environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Phlos/WavePy/HEAD)

## Visualisation functionality
At each stage, visualisation functionality is included. 

### Visualising wave propagation
The wave propagation can be visualised on-the-go by setting `plot_wavefield` to `True` (and including a suitable value for `plot_wavefield_every`):
```python
receivers_out = waveprop.run_waveprop(
    src, rec, model, absorbing_boundaries, 
    plot_wavefield=True, 
    plot_wavefield_every=40, # every x timesteps
    verbose=True)
```

### Visualising seismograms
If (after a forward simulation) a seismogram has been attached to a receiver, this can be visualised with:
```python
rec.plot_seismogram()
```

### Visualising window picks
Equally, window picks (and the resulting adjoint sources) can be visualised with
```pytho
pick = {}
pick['component'] = ['x']   # 'x' or 'z'
pick['times'] = [3.5, 7.5]  # seconds
print('window goes from {} to {} s'.format(pick['times'][0], pick['times'][1]))

receivers_Pwave = waveprop.make_adjoint_source(receivers, pick, plot=3)
```

### Visualising sensitivity kernels
Once sensitivity kernels have been computed, these can be visualised in different parametrisations and as absolute, or relative to a background model. For simple direct waves, it can be instructive to compare the sensitivity kernel to the Fresnel zone. This Fresnel zone can be visualised by setting `plot_Fresnel_zone` to `True`.
```python
kernels..plot_kernels(
    parametrisation='rhovsvp' # could be 'rhomulambda'
    mode='relative',          # could be 'absolute'
    model=model,              # necessary for all kernels other than absolute rhomulambda
    source=src, receiver=receivers[0], 
    plot_Fresnel_zone=True,
)
