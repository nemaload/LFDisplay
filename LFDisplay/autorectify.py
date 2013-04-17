"""
Auto-rectify optimization algorithm.
"""

"""
Parameter space:
        offset-x
        offset-y
        right-dx
        right-dy
        down-dx
        down-dy

Smart X,Y sampling:

(TODO)
Prefer sampling in bright areas.  Compute value histograms per row and column,
use them as empiric probability distributions for X,Y choice (snapping to
nearest grid item), search starting from the center.

Value function:

Sum of @K samples. Each sample is computed as (i) log(#inside) - log(#outside)
or (ii) exp(log(#inside) - log(#outside)) i.e. #inside/#outside.

Optimization:

Approach 1: particle swarm optimization (TODO)

Approach 2: differential evolution (TODO)
"""

import math
import numpy
import random

def autorectify(frame, maxu):
    """
    Automatically detect lenslets in the given frame with the
    given optics parameters (maxu==maxNormalizedSlope) and return
    a tuple of (lensletOffset, lensletHoriz, lensletVert).
    """
    solution = autorectify_de(frame, maxu)
    return solution.to_steps()


def autorectify_de(frame, maxu):
    """
    Autorectification based on differential evolution.
    """
    print "init"
    # Work on a list of solutions, initially random
    solutions_n = 20
    solutions = [RectifyParams([frame.width, frame.height]).randomize()
                     for i in range(solutions_n)]

    image = frame.to_numpy_array()

    print "best..."
    # Initialize the best solution info by something random
    sbest = solutions[0]
    value_best = measure_rectification(image, maxu, sbest)

    # Improve the solutions
    episodes_n = 50
    for e in range(episodes_n):
        print "episode ", e
        # We iterate through the solutions in a random order; this
        # allows us to reuse the same array to obtain recombination
        # solutions randomly and safely.
        permutation = numpy.random.permutation(solutions_n)
        for si in permutation:
            print " solution ", si
            s = solutions[si]
            value_old = measure_rectification(image, maxu, s)
            print "  value ", value_old

            # Cross-over with recombination solutions
            sa = s.to_array()
            r1a = solutions[(si + 1) % solutions_n].to_array()
            r2a = solutions[(si + 2) % solutions_n].to_array()
            r3a = solutions[(si + 3) % solutions_n].to_array()
            dim_k = 5
            CR = 0.5/dim_k
            F = 0.5 * (1 + random.random()) # [0.5,1) random
            co_k = numpy.random.randint(dim_k);
            for k in range(dim_k):
                if k == co_k or random.random() < CR:
                    #print " DE " + str(r1a[k]) + " " + str(r2a[k]) + " " + str(r3a[k])
                    sa[k] = r1a[k] + F * (r2a[k] - r3a[k])

            # Compare and swap if better
            s2 = s
            s2.from_array(sa).normalize()
            value_new = measure_rectification(image, maxu, s2)
            print "  new value ", value_new
            if value_new > value_old:
                solutions[si] = s2
                if value_new > value_best:
                    sbest = s2
                    value_best = value_new

    # Return the best solution encountered
    print "best is ", sbest
    return sbest


def measure_rectification(image, maxu, rparams):
    """
    Measure rectification quality of rparams on a random sample of lens.
    """
    gridsize = rparams.gridsize()
    n_samples = 10 + round(gridsize[0] * gridsize[1] / 400)
    print "  measuring ", rparams, " with grid ", gridsize, " and " ,n_samples ," samples"

    x_coords = numpy.random.randint(int(round(-gridsize[0]/2)), int(round(gridsize[0]/2)), size = n_samples)
    y_coords = numpy.random.randint(int(round(-gridsize[1]/2)), int(round(gridsize[1]/2)), size = n_samples)
    samples = numpy.array([x_coords, y_coords]).T # transpose is like zip!

    value = 0.
    for s in samples:
        value += measure_rectification_one(image, maxu, rparams, s)

    return value / n_samples


def measure_rectification_one(image, maxu, rparams, gridpos):
    """
    Measure rectification of a single given lens.
    """
    lenspos = rparams.xylens(gridpos)
    lenssize = rparams.size * maxu

    # print "measuring ", gridpos, " (", lenspos, ") with ", rparams

    value_inlens = 0.
    value_outlens = 0.

    for x in range(int(round(-rparams.size[0]/2)), int(round(rparams.size[0]/2))):
        for y in range(int(round(-rparams.size[1]/2)), int(round(rparams.size[1]/2))):
            # XXX: we could tilt the whole coordinate grid at once
            # (or actually the source image subspace) to speed things up
            inLensPos = rparams.xytilted([x, y])

            imgpos = lenspos + inLensPos
            # XXX: subpixel sampling?
            imgpos = imgpos.round()

            # print " ", [x, y], " in lens ", rparams, " is ", inLensPos, ", in-img is ", imgpos

            try:
                # sum() is not terribly good pixval, TODO maybe calculate
                # actual brightness?
                pixval = sum(image[int(imgpos[1])][int(imgpos[0])])
            except IndexError:
                # Do not include out-of-canvas pixels in the computation.
                # Therefore, out-of-canvas tiles will have both inlens
                # and outlens values left at zero.
                # print "index error for ", lenspos, " -> ", imgpos
                continue

            # Are we in the ellipsis defined by lenssize?
            if ((inLensPos / lenssize) ** 2).sum() <= 1.:
                value_inlens += pixval
            else:
                value_outlens += pixval
            # print "pixval ", pixval, " for ", lenspos, " -> ", imgpos

    # Just avoid division by zero
    eps = numpy.finfo(numpy.float).eps
    # TODO: Try without log() too
    return math.log((value_inlens + eps) / (value_outlens + eps))


class RectifyParams:
    """
    This class is a container for microlens partitioning information
    (rectification parameters) in a format suitable for optimization:

    framesize[2] (size of the frame; constant)
    size[2] (size of a single lenslet, i.e. the lens grid spacing)
            (but what is evolved is single dimension and aspect ratio)
    offset[2] (shift of the lens grid center relative to image center)
            offset \in [-size/2, +size/2] after normalize()
    tau (tilt of the lens grid, i.e. rotation (CCW) by grid center in radians)
            tau \in [0, pi/8) after normalize()

    (...[2] are numpy arrays)
    """

    def __init__(self, framesize):
        self.framesize = numpy.array(framesize)

    def randomize(self):
        """
        Initialize rectification parameters with random values
        (that would pass normalize() inact).
        """
        # XXX: Something better than uniformly random?
        maxsize = 64
        self.size = numpy.array([0, 0])
        self.size[0] = random.random() * maxsize
        self.size[1] = self.size[0] * (0.8 + random.random() * 0.4)
        self.offset = numpy.array([random.random(), random.random()]) * self.size - self.size/2
        self.tau = random.random() * math.pi/8
        return self

    def gridsize(self):
        """
        Return *approximate* dimensions of the grid defined by an array
        of given lens. Tilt is not taken into account.
        """
        return numpy.array(self.framesize / self.size).round()

    def xytilted(self, ic):
        """
        Return image coordinates tilted by tau.
        """
        return numpy.array([ic[0] * math.cos(self.tau) - ic[1] * math.sin(self.tau),
                            ic[0] * math.sin(self.tau) + ic[1] * math.cos(self.tau)])

    def xylens(self, gc):
        """
        Return image coordinates of a lens at given grid coordinates.
        [0, 0] returns offset[].
        """
        center_pos = self.framesize / 2 + self.offset
        straight_pos = self.size * gc
        tilted_pos = self.xytilted(straight_pos)
        # print "xylens(", gc, ") ", self.framesize, " / 2 = ", self.framesize / 2, " -> ", center_pos, " ... + ", straight_pos, " T", self.tau, " ", tilted_pos, " => ", center_pos + tilted_pos
        return center_pos + tilted_pos

    def normalize(self):
        """
        Normalize parameters so that the offset is by less than
        one lens size (i.e. 0 +- size/2) and tau is less than pi/8.
        """
        self.size = abs(self.size)

        # For <minsize we trim to minsize, but for >maxsize we
        # reset randomly so that our specimen do not cluster
        # around maxsize aimlessly.
        minsize = 5
        maxsize = 64
        if self.size[0] > maxsize:
            self.size[0] = random.random() * maxsize
        elif self.size[0] < minsize:
            self.size[0] = minsize
        if self.size[1] > maxsize:
            self.size[1] = self.size[0] * (0.8 + random.random() * 0.4)
        elif self.size[1] < minsize:
            self.size[1] = minsize

        self.offset = self.offset % self.size - self.size/2
        self.tau = self.tau % (math.pi/8)
        return self

    def to_steps(self):
        """
        Convert parameters to a tuple of
        (lensletOffset, lensletHoriz, lensletVert).
        """
        lensletOffset = self.framesize/2 + self.offset
        lensletHoriz = self.xytilted([self.size[0], 0])
        lensletVert = self.xytilted([0, self.size[1]])
        return (lensletOffset.tolist(), lensletHoriz.tolist(), lensletVert.tolist())

    def to_array(self):
        """
        Convert parameters to an array of values to be opaque
        for optimization algorithm; after optimization pass,
        call from_array to propagate the values back.

        One significant difference between RectifyParams attributes
        and the optimized values is that size is represented not
        as an [x,y] pair but rather [x,aspectratio] pair.
        """
        return numpy.array([self.size[0], float(self.size[1]) / self.size[0], self.offset[0], self.offset[1], self.tau])

    def from_array(self, a):
        """
        Restore parameters from the array serialization.
        """
        self.size[0] = a[0]; self.size[1] = int(a[1] * self.size[0])
        self.offset[0] = a[2]; self.offset[1] = a[3]
        self.tau = a[4]
        return self

    def __str__(self):
        return "[size " + str(self.size) + " offset " + str(self.offset) + " tau " + str(self.tau * 180 / math.pi) + "deg]"
