import h5py
import numpy as np
from sklearn.neighbors import KDTree
import scipy


def open_lookup_table(path):
    '''
    Open the lookup table which can be downloaded from https://zenodo.org/records/6338462#.ZBENinbMKbh

    Parameters

    path (str):
                Location of the file
    
    Returns
    lookup_table (h5py._hl.files.File):
                h5py object containing the full table

    Z_lookup, one dimensional array shape (N,):
                array of stellar metallicities, Z (log10(Z_sun)) which correspond to the first dimension of the spectra array
    
    age_lookup, one dimensional array, shape (M,):
                array of stellar ages (Gyr) which correspond to the second dimension of the spectra array

    
    nu, one dimensional array, shape (L,):
                array of frequencies (Hz) which correspond to the third dimension of the spectra array

    spectra, three dimensional array, shape (N, M, L):
                Corresponding luminosity (L_sun/Hz) for each wavelength, metallicity and age
    
    '''
    lookup_table    = h5py.File(path,'r')
    age_lookup = lookup_table['ages'][:]                #Gyr
    Z_lookup = lookup_table['metallicities'][:]   #log10(Z_sun)
    spectra = lookup_table['spectra'][:]             #L_sun/Hz
    wl           = lookup_table['wavelengths'][:]         #Angstrom
    nu         =  3*10**18/wl    #Hz

    return lookup_table, Z_lookup, age_lookup, nu, spectra

def get_total_L_lookup(spectra, nu, **kwargs):

    '''integrate (using simpson's rule) the Z and age dependent luminosity over frequencies, from nu_min to nu_max if specified, else integrate over all frequencies
    
    Parameters:
    
    spectra, three dimensional array:
            luminosity (L_sun/Hz) for each wavelength, metallicity and age
            First dimension should be Z dependent
            Second dimension should be age dependent
            Third  dimension should be frequency dependent
    
    nu, one dimensional array:
        array of frequencies (Hz) which correspond to the third dimension of the spectra array

    nu_min (optional, float):
        lower integration bound
    
    nu_max (optional, float):
        upper integration bound

    Returns:
    L_lookup, two dimensional array:
        frequency integrated luminosity (relative to solar) for each stellar metallicity and age

    '''

    # unpack kwargs, if integration bounds unspecified take the full range of frequency values
    if 'nu_min'  in kwargs.keys():
        nu_min = kwargs['nu_min']
    else:
        nu_min = nu.min()

    if 'nu_max' in kwargs.keys():
        nu_max = kwargs['nu_max']
    else:
        nu_max= nu.max()

    
    # integration requires nu to be increasing so sort values
    sorter = np.argsort(nu)

    nu = nu[sorter]
    spectra = spectra[:,:, sorter]
    # find frequency values within bounds
    nu_mask =(nu > nu_min )* (nu<nu_max)



    # integrate over frequencies (last dimension)
    L_lookup = scipy.integrate.simpson(spectra[:,:, nu_mask],x=nu[nu_mask], axis=-1) # Units are L_sun

    return L_lookup

def find_closest_quantity_id(lookup, values):
    '''
    Find index of closest match for value in lookup

    lookup, array:
        the array in which to find the closest match for each value
    
    values, array: 
        the values for which to find index of closest match within lookup

    Return
    id, array, shape (N,):
        index of lookup closest to value

    '''
    # create kd tree, cityblock metric
    tree = KDTree(lookup.reshape(-1,1), leaf_size=2, metric='cityblock')

    # query for closest match     
    id = tree.query(values.reshape(-1, 1), k=1)[1].flatten()

    return id

def match_star_luminosities(Z, t, Z_lookup, t_lookup, L_lookup):
    ''' 
    get the luminosity of each star of given Z,t from Z and age dependent luminosity

    Parameters
    Z, array, shape (K, ):
        metallicity (log10(Z_sun)) of each stellar pop to find L
    
    t, array, shape (K, ):
        age (Gyr) of each stellar pop to find L

    Z_lookup, one dimensional array shape (N,):
        array of stellar metallicities, Z (log10(Z_sun)) which correspond to the first dimension of the spectra array
    
    age_lookup, one dimensional array, shape (M,):
        array of stellar ages (Gyr) which correspond to the second dimension of the spectra array
    
    L_lookup, two dimensional array, shape (N,M):
        frequency integrated luminosity (relative to solar) for each stellar metallicity and age

    Returns
    L, array, shape (K,):
        Luminosity of each stellar pop

    

    '''
    # index of closest age match in L_lookup
    t_id = find_closest_quantity_id(t_lookup, t)
    # index of closest Z match in L_lookup 
    Z_id = find_closest_quantity_id(Z_lookup, Z)

    L = L_lookup[Z_id, t_id] # L_sun

    return L

def find_nearest_stars(gas_coords, star_coords, N_star=None, leafsize=32):
    '''
    find the closest N_star with locations in star_coords for each gas cell with locations found in gas_coords

    Parameters
    gas_coords, array (M, 3):
        three dimensional location of each gas cell
    
    star_coords, array (K, 3)
        three dimensional location of each star particle

    N_star (optional), int:
        Number of stars to find for each gas cell

    leafsize (optional), int:
        as specified at https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
    
    Returns
    dist, array, shape (M, N_star):
        distance to the closest N_star for each gas cell
    
    ind, array, shape (M, N_star):
        index within star_coords of the closest N_star to each gas cell

    '''

    # if N_star isn't specified, use all of the stars
    if N_star is None:
        N_star = len(star_coords)
    
    # create kd tree
    tree = KDTree(star_coords, leaf_size=leafsize)

    # query tree for closest N_stars
    dist, ind = tree.query(gas_coords.reshape(-1,3), k = N_star) # 3d distance in kpc


    return dist, ind


def illuminate_gas(L_star, distance, index, return_sum=True):
    '''
    get the gas illumination (?) for a gas cell from the distance to the closest N_stars, their luminosities and the index to find the luminosities

    Parameters
    L_star, array, shape (K,):
        array of luminosities for all stars

    distance, array, shape (M, N):
        distance to N nearest stars for each of the M gas cell
    index, array, shape (M, N):
        index of N nearest stars within L_star for each of the M gas cell
    
    return_sum, default=True:
        whether to return the sum of each of the contributions to the illumination(?)
        if False, the individual contributions from each of the N stars is returned
    
    Returns
    gas_illumination, array, shape (M,):
        the total illumination(?) of each gas cell if return_sum=True
    
    gas_illumination_contributions, array, shape (M, N):
        the contribution of each N star to the illumination for each gas cell
    '''

    gas_illumination_contributions = L_star[index]/(4*np.pi*distance**2)
    
    gas_illumination = gas_illumination_contributions.sum(axis=-1)

    if return_sum:
        return gas_illumination
    
    else:
        return gas_illumination_contributions
    
def project_quantity(coords, quantity, N_px=100, dims=[0, 1]):
    '''
    Project a quantity from three dimensions into two dimensions

    Parameters
    coords, array, shape (N,3):
        coordinates of the gas cells in three dimensions
    
    
    quantity, array, shape (N,):
        the quantity to project for each gas cell

    N_px, int:
        the number of pixels along each dimension of the projection. default= 100
    
    dims, listlike, length 2:
        the spatial dimensions to include in the projection

    Returns
    projected_quantity, array, shape (N_px, N_px):
        two dimensional projection of the quantity
    
    xs, array, shape (N_px+1,):
        The bin edges along the first dimension
    
    ys, array, shape (N_px+1,):
        The bin edges along the second dimension

    '''
    projected_quantity, xs, ys = np.histogram2d(coords[:,dims[0]], coords[:,dims[1]], weights=quantity, bins=N_px)
    return projected_quantity.T, xs, ys

def project_density(coords, quantity, N_px=100):
    '''
    Project a quantity from three dimensions into two dimensionsional density

    Parameters
    coords, array, shape (N,3):
        coordinates of the gas cells in three dimensions
    
    
    quantity, array, shape (N,):
        the quantity to project for each gas cell (eg. mass)

    N_px, int:
        the number of pixels along each dimension of the projection. default= 100
    
    dims, listlike, length 2:
        the spatial dimensions to include in the projection

    Returns
    projected_quantity_density, array, shape (N_px, N_px):
        two dimensional density projection of the quantity
    
    xs, array, shape (N_px+1,):
        The bin edges along the first dimension
    
    ys, array, shape (N_px+1,):
        The bin edges along the second dimension

    '''
    density, xs, ys = np.histogram2d(coords[:,0], coords[:,1], weights=quantity, bins=N_px)
    density /= (np.diff(xs)[0]*np.diff(ys)[0])
    return density.T, xs, ys






