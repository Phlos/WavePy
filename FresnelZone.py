# all the fresnel zone stuff

import numpy as np

def Fresnel_width_from_velocity_field_and_stf(
    velocity_field: np.ndarray,
    # source: Source,
    # receiver: Receiver,
    loc_s: np.ndarray, loc_r: np.ndarray,
    time_axis: np.ndarray, stf: np.ndarray,
    verbose: bool=True,
):
    '''
    convenience function to compute the width of a Fresnel
    zone based on the average velocity within a (2-D) 
    supplied velocity field, and a source and receiver 
    location.
    '''

    v = velocity_field.mean()
    if verbose:
        print('Seismic velocity: {} km/s'.format(v/1000.))
    
    dt = time_axis[1] - time_axis[0]
    L = np.abs(loc_r[0] - loc_s[0])
    if verbose:
        print('Source-receiver distance = {} km'.format(L/1000.))
    # stf = src['stf']['x']      ### TODO don't hardcode x component!
    # stf = source.stf_directionless
    spec = np.fft.fft(stf) # fast fourier transform
    freq = np.fft.fftfreq(spec.size, d = dt / 4.) # frequency axis
    f = freq[np.argmax(spec)]
    if verbose:
        print('Largest amplitude frequency in source spectrum: {:.2f} Hz'.format(f))
    w_f = Fresnel_width(v, L, f)
    if verbose:
        print('==> Fresnel zone width is {:.2f} km'.format(w_f/1000.))
    
    return w_f


def Fresnel_width(v, L, f):
    
    return np.sqrt((v*L)/f)

def plot_Fresnel(
    ax, loc_s, loc_r, w_f, 
    units='km',
    **kwargs):
    
    f = 1./1000. if units=='km' else 1.
    xx,yy, a,c,h,v = ellipse_from_f1_f2_b(
        loc_s*f, loc_r*f, w_f*f)
    
    # print('x: {:.2f} - {:.2f}    y: {:.2f} - {:.2f}'.format(min(xx), max(xx), min(yy), max(yy)))
    ax.plot(xx,yy, **kwargs)
    

def half_ellipse(a, b, h=0, v=0):
    
    # determine x axis
    xmin = h-a
    xmax = h+a
    x = np.linspace(xmin, xmax, 500)
    
    stuff = b**2 * (1 - (x-h)**2/a**2)
    
    # mask out accidental negative values (avoid sqrt of negative)
    x = x[stuff >= 0]
    stuff = stuff[stuff >= 0]

    # determine y
    y = np.sqrt(stuff) + v
    # determine c
    c = np.sqrt(a**2 - b**2)
    
    f1 = (-c+h, v)
    f2 = (c+h, v)
    
    return x, y, c, f1, f2


def ellipse(a, b, h=0, v=0):
    
    x, y, c, f1, f2 = half_ellipse(a,b,h,v)
    xx = np.concatenate((x,x[::-1]))
    yy = np.concatenate((y,y[::-1]*-1+2*v)) #  the 2*v term counteracts the vertical shift 
                                            #+ by v for the bottom half

    return xx,yy,c,f1,f2


def ellipse_from_f1_f2_b(f1, f2, b):
    
    v = f1[1]
    h = 0.5 * (f1[0] + f2[0])
    c = 0.5 * (f2[0] - f1[0])
    
    a = np.sqrt(c**2 + b**2)
    
    xx,yy, _, _, _ = ellipse(a, b, h=h, v=v)
    
    return xx, yy, a, c, h, v
