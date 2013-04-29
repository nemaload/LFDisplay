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

import cv
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches


MAX_RADIUS = 30


def autorectify(frame, maxu):
    """
    Automatically detect lenslets in the given frame with the
    given optics parameters (maxu==maxNormalizedSlope) and return
    a tuple of (lensletOffset, lensletHoriz, lensletVert).
    """
    # solution = autorectify_de(frame, maxu)
    solution = autorectify_cv(frame, maxu)
    return solution.to_steps()


def autorectify_cv(frame, maxu):
    """
    Autorectification based on computer vision analysis.
    """
    image = frame.to_numpy_array()

    tiling = ImageTiling(image, MAX_RADIUS * 5)
    tiling.scan_brightness()

    fftsumpow = numpy.zeros([tiling.tile_step, tiling.tile_step])

    n_samples = 2
    tiles = range(n_samples)
    colors = [ "lightsalmon", "lightgreen", "lightblue", "red", "green", "blue" ]
    for i in range(n_samples):
        t = tiling.random_tile()
        (ul, br) = tiling.tile_to_imgxy(t)
        tiles[i] = (ul, br)
        s = tiling.tile_step
        # image pixels in the chosen tile
        timage = TileImage(tiling, image[ul[1]:br[1], ul[0]:br[0]].reshape(s, s).copy())
        timage.to256()

        if 1:
            timage.threshold(maxu).blur()

        # 2D Hanning window
        if 0:
            window = timage.hanning_outer_window()
        else:
            window = timage.hanning_rotated_window()
        twindowed = TileImage(tiling, window * timage.image)

        # Show the windowed tile
        print("tile " + str(i) + ": " + colors[i])
        print twindowed.image
        plt.figure("Wind. tile " + str(i) + ": " + colors[i])
        imgplot = plt.imshow(twindowed.image, cmap=plt.cm.gray)
        plt.show()

        # 2D FFT
        fftspectrum = TileSpectrum(numpy.fft.fft2(twindowed.image))

        # Show the power spectrum of the tile
        fftpow = fftspectrum.power()
        fftsumpow += fftpow
        plt.figure("FFT tile " + str(i) + ": " + colors[i])
        plt.imshow(numpy.log(fftpow))
        plt.show()

        # Filter the spectrum
        fftcropspectrum = fftspectrum.bandpass(16, 16)

        # Show the filtered power spectrum
        plt.figure("FFT croptile " + str(i) + ": " + colors[i])
        plt.imshow(numpy.log(fftcropspectrum.power()))
        plt.show()
        # Show the original image reconstructed back from the filtered
        # frequency data to demonstrate we are on to something :)
        plt.figure("IFFT croptile " + str(i) + ": " + colors[i])
        plt.imshow(numpy.real(numpy.fft.ifft2(fftcropspectrum.spectrum)), cmap=plt.cm.gray)
        plt.show()

        # 2. ???
        # 3. PROFIT!

    # Show window with whole image, tile parts highlighted
    f = plt.figure("whole")
    imgplot = plt.imshow(image.reshape(frame.height, frame.width), cmap = plt.cm.gray)
    for i in range(n_samples):
        (ul, br) = tiles[i]
        ax = f.add_subplot(111)
        rect = matplotlib.patches.Rectangle((ul[0],ul[1]),
                width=tiling.tile_step, height=tiling.tile_step,
                edgecolor=colors[i], fill=0)
        ax.add_patch(rect)
    plt.show()

    #plt.figure("FFT power sum over all scanned tiles")
    #plt.imshow(numpy.log(fftsumpow))
    #plt.show()

    # XXX: We just return random parameters for now; this method
    # is not finished.
    return RectifyParams([frame.width, frame.height]).randomize()


class TileImage:
    """
    A holding class for image (numpy array) of a single tile
    analyzed.
    """
    def __init__(self, tiling, timage):
        self.tiling = tiling
        self.image = timage
    def to256(self):
        self.image = 255. - self.image.astype('float') * 255. / self.image.max()
        return self

    def threshold(self, maxu):
        # Threshold such that background is black, foreground is white
        # (you may want to turn this on/off based on the method below)
        background_color = self.tiling.background_color(self.image, maxu)
        foreground_i = self.image > background_color
        self.image[foreground_i] = 1.
        self.image[numpy.invert(foreground_i)] = 0.
        return self
    def blur(self):
        # FFT hates sharp edges, so blur us; however, bad blurring
        # would introduce artificial frequencies; rectangular uniform
        # blur seems like exactly what we want, though
        self.image = cv2.blur(self.image, (MAX_RADIUS,MAX_RADIUS))
        #self.image = cv2.GaussianBlur(self.image, (MAX_RADIUS-1,MAX_RADIUS-1), 0)
        return self

    def hanning_outer_window(self):
        # ...outer product way
        window = numpy.hanning(self.tiling.tile_step)
        window = numpy.outer(window, window)
        return window
    def hanning_rotated_window(self):
        # ...rotational way
        # W(x,y) = 0.5 + 0.5 * cos(pi * r(x,y)/r(max))
        s = self.tiling.tile_step
        ri = numpy.linspace(-s/2, s/2, s)
        rx = numpy.repeat(ri[numpy.newaxis,:], s, 0)
        ry = numpy.repeat(ri[:,numpy.newaxis], s, 1)
        r = numpy.sqrt(rx * rx + ry * ry)
        rmax = r[int(s/2),0]
        numpy.clip(r, 0., rmax, out=r)
        window = 0.5 * numpy.cos(math.pi * r/rmax) + 0.5
        return window

class TileSpectrum:
    """
    A holding class for 2D frequency spectrum (numpy array)
    of a single tile analyzed.
    """
    def __init__(self, spectrum):
        self.spectrum = spectrum

    def bandpass(self, cx, cy):
        sx = self.spectrum.shape[1]
        sy = self.spectrum.shape[0]

        # Low-pass filter - keep only 16x16 corners of the original
        # array, therefore keeping only the low frequency component
        fftcropmask = numpy.logical_not(numpy.zeros([sy,sx]).astype(bool))
        fftcropmask[:,cx:sx-cx] = False
        fftcropmask[cy:sy-cy] = False
        # Delete the central rows + columns of fftcropimage
        fftcropspectrum = self.spectrum.copy()[fftcropmask].reshape(cy*2,cx*2)

        # An awful "high-pass" filter that should just punch out
        # the DC and window component
        fftcropspectrum[0:2,0:2] = 0.001
        fftcropspectrum[0:2,cx*2-2:cx*2] = 0.001
        fftcropspectrum[cy*2-2:cy*2,0:2] = 0.001
        fftcropspectrum[cy*2-2:cy*2,cx*2-2:cx*2] = 0.001

        return TileSpectrum(fftcropspectrum)

    def power(self):
        return numpy.real(numpy.multiply(self.spectrum, self.spectrum.conjugate()))


def autorectify_de(frame, maxu):
    """
    Autorectification based on differential evolution.
    """
    print "init"
    # Work on a list of solutions, initially random
    solutions_n = 20
    solutions = [RectifyParams([frame.width, frame.height]).randomize()
                     for i in range(solutions_n)]
    # 2160x2560 -> 1080x1280
    # (1284.367000,1190.300000,23.299000,-0.032000,-0.035000,23.262000)
    # (x-offset,y-offset,right-dx,right-dy,down-dx,down-dy)
    solutions[0].size[0] = 23.3
    solutions[0].size[1] = 23.3
    solutions[0].tau = 0.0137397938
    solutions[0].offset[0] = -8.907
    solutions[0].offset[1] = -6.303
    print solutions[0].to_steps()

    image = frame.to_numpy_array()
    tiling = ImageTiling(image, MAX_RADIUS * 5)
    tiling.scan_brightness()

    print "best..."
    # Initialize the best solution info by something random
    sbest = solutions[0]
    value_best = measure_rectification(image, tiling, maxu, sbest)

    # Improve the solutions
    episodes_n = 50
    for e in range(episodes_n):
        print "EPISODE ", e
        # We iterate through the solutions in a random order; this
        # allows us to reuse the same array to obtain recombination
        # solutions randomly and safely.
        permutation = numpy.random.permutation(solutions_n)
        for si in permutation:
            print " solution ", si
            s = solutions[si]
            value_old = measure_rectification(image, tiling, maxu, s)
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
            value_new = measure_rectification(image, tiling, maxu, s2)
            print "  new value ", value_new
            if value_new > value_old:
                print "   ...better than before"
                solutions[si] = s2
                if value_new > value_best:
                    print "   ...and best so far!"
                    sbest = s2
                    value_best = value_new

    # Return the best solution encountered
    print "best is ", sbest, " with value ", value_best
    return sbest


def measure_rectification(image, tiling, maxu, rparams):
    """
    Measure rectification quality of rparams on a random sample of lens.
    """
    gridsize = rparams.gridsize()
    n_samples = int(10 + round(gridsize[0] * gridsize[1] / 400))
    print "  measuring ", rparams, " with grid ", gridsize, " and " ,n_samples ," samples"

    value = 0.

    # TODO: Draw all samples at once
    for i in range(n_samples):
        t = tiling.random_tile()
        s = tiling.tile_to_lens(t, rparams)
        # print "tile ", t, "lens ", s
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
                # print " IN LENS pixval ", pixval, " for ", lenspos, " -> ", imgpos
                value_inlens += pixval
            else:
                # print "OUT LENS pixval ", pixval, " for ", lenspos, " -> ", imgpos
                value_outlens += pixval

    # Just avoid division by zero
    eps = numpy.finfo(numpy.float).eps
    return (value_inlens + eps) / (value_outlens + eps)


class RectifyParams:
    """
    This class is a container for microlens partitioning information
    (rectification parameters) in a format suitable for optimization:

    framesize[2] (size of the frame; constant)
    size[2] (size of a single lenslet, i.e. the lens grid spacing)
            (but what is evolved is single dimension and aspect ratio)
            size \in [5, 64] after normalize()
    offset[2] (shift of the lens grid center relative to image center)
            offset \in [-size/2, +size/2] after normalize()
    tau (tilt of the lens grid, i.e. rotation (CCW) by grid center in radians)
            tau \in [0, pi/8) after normalize()

    (...[2] are numpy arrays)
    """

    def __init__(self, framesize):
        self.framesize = numpy.array(framesize)
        self.minsize = 12
        self.maxsize = MAX_RADIUS

    def randomize(self):
        """
        Initialize rectification parameters with random values
        (that would pass normalize() inact).
        """
        # XXX: Something better than uniformly random?
        self.size = numpy.array([0., 0.])
        self.size[0] = self.minsize + random.random() * (self.maxsize - self.minsize)
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

    def xytilted_tau(self, ic, tau):
        """
        Return image coordinates tilted by given tau.
        """
        return numpy.array([ic[0] * math.cos(tau) - ic[1] * math.sin(tau),
                            ic[0] * math.sin(tau) + ic[1] * math.cos(tau)])

    def xytilted(self, ic):
        """
        Return image coordinates tilted by self.tau.
        """
        return self.xytilted_tau(ic, self.tau)

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

    def lensxy(self, ic):
        """
        Return lens grid coordinates corresponding to given image coordinates.
        """
        center_pos = self.framesize / 2 + self.offset
        tilted_pos = ic - center_pos
        straight_pos = self.xytilted_tau(tilted_pos, -self.tau)
        return (straight_pos / self.size).astype(int)

    def normalize(self):
        """
        Normalize parameters so that the offset is by less than
        one lens size (i.e. 0 +- size/2) and tau is less than pi/8.
        """
        self.size = abs(self.size)

        # For <minsize we trim to minsize, but for >maxsize we
        # reset randomly so that our specimen do not cluster
        # around maxsize aimlessly.
        if self.size[0] > self.maxsize:
            self.size[0] = self.minsize + random.random() * (self.maxsize - self.minsize)
        elif self.size[0] < self.minsize:
            self.size[0] = self.minsize
        if self.size[1] > self.maxsize:
            self.size[1] = self.size[0] * (0.8 + random.random() * 0.4)
        elif self.size[1] < self.minsize:
            self.size[1] = self.minsize

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
        self.size[0] = a[0]; self.size[1] = a[1] * self.size[0]
        self.offset[0] = a[2]; self.offset[1] = a[3]
        self.tau = a[4]
        return self

    def __str__(self):
        return "[size " + str(self.size) + " offset " + str(self.offset) + " tau " + str(self.tau * 180 / math.pi) + "deg]"


class ImageTiling:
    """
    This class represents a certain tiling of the image (numpy 3D array)
    used for brightness data collection and sampling.
    """

    def __init__(self, image, tile_step):
        self.tile_step = tile_step
        self.height_t = int(image.shape[0] / tile_step)
        self.width_t = int(image.shape[1] / tile_step)

        # Adjust image view to be cropped on tile boundary
        self.height = self.height_t * tile_step
        self.width = self.width_t * tile_step
        self.image = image[0:self.height, 0:self.width]

    def scan_brightness(self):
        """
        Compute per-tile brightness data (mean, sd)
        and construct an empiric probability distribution based
        on this data (...that prefers tiles with average brightness).
        """

        # Create a brightness map from image
        # sum() is not terribly good brightness approximation
        brightmap = self.image.sum(2)

        # Group rows and columns by tiles
        tiledmap = brightmap.reshape([self.height_t, self.tile_step, self.width_t, self.tile_step])

        # brightavgtiles is brightness mean of specific tiles
        self.brightavgtiles = tiledmap.mean(3).mean(1)
        # brightstdtiles is brightness S.D. of specific tiles
        #self.brightstdtiles = numpy.sqrt(tiledmap.var(3).mean(1))

        # rescale per-tile brightness mean so that minimum is 0
        # and maximum is 1
        minbrightness = self.brightavgtiles.min()
        maxbrightness = self.brightavgtiles.max()
        ptpbrightness = maxbrightness - minbrightness
        brightxavgtiles = (self.brightavgtiles - minbrightness) / ptpbrightness

        # construct probability distribution such that
        # xavg 0.5 has highest probability
        # TODO: Also consider S.D.? But how exactly?
        # We might want to maximize S.D. to focus on
        # areas with sharpest lens shapes, or minimize S.D.
        # to focus on areas with most uniform lens interior...
        # TODO: Nicer distribution shape?
        self.pdtiles = 0.5*0.5*0.5 - numpy.power(0.5 - brightxavgtiles, 3)
        self.pdtiles_sum = self.pdtiles.sum()

        #for t in numpy.mgrid[0:self.height_t, 0:self.width_t].T.reshape(self.height_t * self.width_t, 2):
        #    # t = [y,x] tile index
        #    print(t[1], " ", t[0], ": ", self.pdtiles[t[0], t[1]], " (bavg ", self.brightavgtiles[t[0], t[1]], " bxavg ", brightxavgtiles[t[0], t[1]], ")")

        return self

    def random_tile(self):
        """
        Choose a random tile with regards to the brightness distribution
        among tiles (as pre-processed by scan_brightness().
        """

        stab = random.random() * self.pdtiles_sum
        for t in numpy.mgrid[0:self.height_t, 0:self.width_t].T.reshape(self.height_t * self.width_t, 2):
            # t = [y,x] tile index
            prob = self.pdtiles[t[0], t[1]]
            if prob > stab:
                return numpy.array([t[1], t[0]])
            stab -= prob
        # We reach here only in case of float arithmetic imprecisions;
        # just pick a uniformly random tile
        print "ImageTiling.random_tile(): fallback to random (warning)"
        return numpy.array([numpy.random.randint(self.height_t), numpy.random.randint(self.width_t)])

    def tile_to_imgxy(self, tile):
        """
        Return image coordinates corresponding to the top left
        and bottom right corner of a given tile.
        """
        return (numpy.array([tile[0] * self.tile_step,
                             tile[1] * self.tile_step]),
                numpy.array([(tile[0]+1) * self.tile_step,
                             (tile[1]+1) * self.tile_step]))

    def tile_to_lens(self, tile, rparams):
        """
        Return lens grid coordinates corresponding to the center
        of a given tile.
        """
        imgcoords = numpy.array([
            tile[0] * self.tile_step + self.tile_step/2,
            tile[1] * self.tile_step + self.tile_step/2])
        return rparams.lensxy(imgcoords)

    def background_color(self, timage, maxu):
        """
        Make an educated guess on the shade level corresponding to the
        background (non-lens) color.
        """
        # The lens radius shall correspond to maxu * grid step.
        # Therefore, lens array area is 2*maxu * total_area and
        # 1-lens_area shall be background color. In addition,
        # in the lens array area, single lens square is 4*r^2
        # while the lens circle is pi*r^2.
        # Therefore, background color should occupy roughly
        # (1-2*maxu + 1-pi/4) fraction of the whole area.
        background_fract = (1 - 2*maxu) + (1 - math.pi/4)

        # color_relcounts will contain relative counts of all
        # shades in timage
        color_nshades = 256
        (color_counts, color_bounds) = numpy.histogram(timage, color_nshades, (0., timage.max()))
        color_relcounts = color_counts.astype('float32') / float(self.tile_step ** 2)
        #print timage
        #print color_counts
        #print color_relcounts

        # find the boundary nearest to background_fract
        sum_fract = 0.
        for i in range(color_nshades):
            #print "bg " + str(background_fract) + " i " + str(i) + " rc " + str(color_relcounts[i]) + " +> " + str(sum_fract)
            sum_fract_2 = sum_fract + color_relcounts[i]
            if sum_fract_2 > background_fract:
                # good, we are over the boundary! but maybe it's
                # a closer shot if we choose the previous shade
                if i > 0 and background_fract - sum_fract < sum_fract_2 - background_fract:
                    return (color_bounds[i-1] + color_bounds[i]) / 2
                elif i < color_nshades-1:
                    return (color_bounds[i+1] + color_bounds[i]) / 2
                else:
                    return color_bounds[i-1]
            sum_fract = sum_fract_2
        return 0.5 # unf, what else to do?
