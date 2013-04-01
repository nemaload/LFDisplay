"Qt style attribute builder"

attributes = [
    ('ApertureCenter','apertureCenter'),
    ('RTMFocus','rtmFocus'),
    ('RTMPerspective','rtmPerspective'),
    ('ApertureDiameter','apertureDiameter'),
    ('DisplayType','displayType'),
    ('ZoomPower','zoomPower'),
    ('BackgroundColor','backgroundColor'),
    ('Gain','gain'),
    ('Offset','offset'),
    ('Gamma','gamma'),
    ('IntensityCorrection','intensityCorrection'),
    ('LensletOffset','lensletOffset'),
    ('LensletHorizontal','lensletHorizontal'),
    ('LensletVertical','lensletVertical'),
    ('DrawGrid','drawGrid'),
    ('GridType','gridType'),
    ('GridCentersColor','gridCentersColor'),
    ('GridEdgesColor','gridEdgesColor'),
    ('InputQueue','inputQueue'),
    ('DesiredIntensity','desiredIntensity'),
    ('ImageMatrix','imageMatrix'),
    ]

def setter(capname, name):
    s="""
    def setATTRIBUTE_UPPER(self, attribute_lower, doSave=True):
        self._lock.lock()
        oldATTRIBUTE_UPPER = self._attribute_lower
        self._attribute_lower = attribute_lower
        self._lock.unlock()
        if oldATTRIBUTE_UPPER != attribute_lower:
            self.emit(QtCore.SIGNAL('attribute_lowerChanged()'))
            if doSave:
                self.saveSettings()"""
    s=s.replace("ATTRIBUTE_UPPER",capname).replace("attribute_lower",name)
    return s

def getter(capname, name):
    s="""
    def attribute_lower(self):
        self._lock.lock()
        attribute_lower = self._attribute_lower
        self._lock.unlock()
        return attribute_lower"""
    s=s.replace("ATTRIBUTE_UPPER",capname).replace("attribute_lower",name)
    return s

print '    # BEGIN GENERATED METHODS'
for capname, name in attributes:
    print setter(capname, name)
    print getter(capname, name)
print
print '    # END GENERATED METHODS'

    
