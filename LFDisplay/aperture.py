"""
Aperture sampling pattern generators
"""
import math
import random

class Error(Exception):
    pass

class ApertureSampler:
    def __init__(self, center, max_bounds):
        """
        Initialize the sampler

        center is a tuple (u,v) that specifies
        the center of the aperture

        max_bounds is a tuple of (left,right,bottom,top)
        which specifies the maximum bounds of the sample
        grid

        For example, max_bounds=(-0.5,0.5,-0.5,0.5) will generate
        sampling patterns that cover an area that is at
        most 1.0 units wide and 1.0 units high

        The bounds are offsets from the center
        """
        self.center = tuple(center)
        self.max_bounds = tuple(max_bounds)

    def get(self, aperture_func=lambda u,v: True):
        """
        Return a list of tuples of the form (u,v,weight)
        corresponding to the aperture sampling locations
        and weights.

        aperture_func can be a function u,v -> bool which
        returns false if (u,v) is outside of a custom aperture
        shape
        """
        raise Error('Unimplemented')

class GridApertureSampler(ApertureSampler):
    """
    Create a uniform grid sampling pattern
    """
    def __init__(self, center, max_bounds, spacing):
        """
        Create a uniform grid sampling pattern

        spacing is a tuple (du,dv) that specifies the distance
        between two adjacent points in u or in v
        """
        ApertureSampler.__init__(self, center, max_bounds)
        self.spacing = tuple(spacing)
        self.center = center
        # generate our pattern
        u_min = (int(self.max_bounds[0]/self.spacing[0]) + 0.5)*self.spacing[0]
        v_min = (int(self.max_bounds[2]/self.spacing[1]) + 0.5)*self.spacing[1]
        u_num = int((self.max_bounds[1]-self.max_bounds[0])/self.spacing[0])
        v_num = int((self.max_bounds[3]-self.max_bounds[2])/self.spacing[1])
        # correct for the center
        cur_center_u = (u_num-1)*0.5*self.spacing[0] + u_min
        cur_center_v = (v_num-1)*0.5*self.spacing[1] + v_min
        u_min = u_min + self.center[0] - cur_center_u
        v_min = v_min + self.center[1] - cur_center_v
        # generate the samples
        u_samples = [u*self.spacing[0] + u_min for u in range(u_num)]
        v_samples = [v*self.spacing[0] + v_min for v in range(v_num)]
        # calculate a weight
        self.weight = 1./(u_num*v_num)
        # repeat and create a Cartesian product
        u_rep = [u for u in u_samples for i in range(v_num)]
        v_rep = v_samples * u_num
        self.uv_samples = zip(u_rep,v_rep)
        # print self.uv_samples

    def get(self, aperture_func=lambda u,v: True):
        matched_samples = [(u,v) for u,v in self.uv_samples if aperture_func(u,v)]
        if matched_samples:
            weight = 1./len(matched_samples)
            samples = [(u,v,weight) for u,v in matched_samples]
            return samples
        else:
            return [(self.center[0],self.center[1],1.0)]

class JitteredGridApertureSampler(GridApertureSampler):
    """
    Create a jittered uniform grid sampling pattern
    """
    def __init__(self, center, max_bounds, spacing, jitter=(0.5,0.5)):
        """
        Create a uniform grid sampling pattern

        spacing is a tuple (du,dv) that specifies the distance
        between two adjacent points in u or in v

        jitter is a tuple (maxu,maxv) that specifies the maximum
        jitter proportional to grid spacing
        The default (0.5,0.5), allows for the maximum jitter
        """
        from random import random
        # create the grid
        GridApertureSampler.__init__(self, center, max_bounds, spacing)
        self.jitter = jitter
        # now jitter it
        scale = (self.jitter[0]*2*self.spacing[0],
                 self.jitter[1]*2*self.spacing[1])
        offset = (-self.jitter[0]*self.spacing[0],
                  -self.jitter[1]*self.spacing[1])
        self.uv_samples = [(u+scale[0]*random()+offset[0],
                            v+scale[1]*random()+offset[1]) for u,v in self.uv_samples]

    def get(self, aperture_func=lambda u,v: True):
        matched_samples = [(u,v) for u,v in self.uv_samples if aperture_func(u,v)]
        if matched_samples:
            weight = 1./len(matched_samples)
            samples = [(u,v,weight) for u,v in matched_samples]
            return samples
        else:
            return [(self.center[0],self.center[1],1.0)]

def getCircularAperture(num_samples, jitter=True):
    """
    Return a set of sample points in a circular aperture of radius 0.5
    """
    # create a square grid of samples that has an odd number
    # samples and has enough samples to enclose a circle
    # of num_samples samples with the same sampling density
    width = int(math.ceil((num_samples * 4.0 / math.pi)**0.5))
    if width%2 == 0:
        width = width + 1
    linear_samples = [(0.5+x)/width-0.5 for x in range(width)]
    u_rep = [u for u in linear_samples for i in range(width)]
    v_rep = linear_samples * width
    samples = zip(u_rep,v_rep)
    # jitter all the samples and recenter
    if jitter:
        center = int((width*width-1)/2)
        jitter_range = ((0.5*0.5*0.5)**0.5)/width
        samples = [(x+random.uniform(-jitter_range,jitter_range),
                    y+random.uniform(-jitter_range,jitter_range))
                   for (x,y) in samples]
        samples = [(x-samples[center][0],
                    y-samples[center][1]) for (x,y) in samples]

    # pick the num_samples closest points to the center
    def distance(coordinate):
        x,y = coordinate
        return x*x+y*y
    samples = sorted(samples, key=distance)
    samples = samples[0:num_samples]

    # rescale so that we have an apparent radius of 0.5
    rescale = 1.0/(2*(num_samples / math.pi)**0.5) / (1.0/width)
    max_radius = distance(samples[-1])**0.5
    if max_radius*rescale > 0.5:
        rescale = 0.5/max_radius

    weight = 1.0/num_samples
    samples = [(x*rescale,y*rescale,weight) for (x,y) in samples]

    return samples

