#==========================================================================================================#
# timescales.PY
#==========================================================================================================#

#!/opt/local/bin/python

# A collection of subroutines to calculate astronomically important timescales

#==========================================================================================================#

# Import modules

import numpy as np

# Global parameters

G            = 6.67408e-11
mh           = 1.6737236e-27 # Mass of a hydrogen atom in kg
msun         = 1.989e30 # Mass of sun in kg
pc           = 3.08567758e16 # A parsec in metres
percm2perm   = 1.e6
sin1yr       = 3.15569e7

# Define subroutines

def tff_spherical( number_density, mu ):

    # Accepts a number density in units of particles per cubic centimetre and derives the free fall time in yrs

    mass_density = mu * mh * number_density * percm2perm

    tff = np.sqrt( (3. * np.pi) / (32. * G * mass_density) )

    tff = tff / sin1yr # free-fall time in years

    return tff
