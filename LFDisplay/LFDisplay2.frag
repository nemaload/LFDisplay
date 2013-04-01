uniform sampler2D tex;
uniform vec4 gain;
uniform vec4 offset;
uniform vec4 gamma;
uniform mat2 rectLinear;
uniform vec2 rectOffset;
uniform vec4 normalDim;
uniform mat4 RTM;
uniform sampler2D aptex; // aperture texture
uniform vec2 apertureScale; // how much scaling to multiply for aperture

void main(void) {
  vec2 coord, coord0, coord1, coord2, coord3, coord4;
  vec2 uv;
  vec4 fracts;
  vec4 xyuv;
  vec4 aperture;
  int x, y;
  vec4 color;

  color = vec4(0.0,0.0,0.0,1.0);

  for(y = 0 ; y < %d ; y += 1) {
    for(x = 0 ; x < %d ; x += 1) {
      aperture = texture2D(aptex,vec2((float(x)+0.5)/%d.0,(float(y)+0.5)/%d.0));
      // comment: multiplying by 4th channel in aperture to obtain proper scaling and ignoring aperture component in input coordinate for abbe sine condition reasons
      xyuv = RTM*vec4(gl_TexCoord[0].xy, apertureScale*aperture.w*(aperture.xy-vec2(0.5,0.5)));
      if(aperture.z == 0.0) break;
      if(all(bvec4(lessThan(abs(xyuv.pq),vec2(0.5,0.5)),lessThan(abs(xyuv.st),vec2(0.5,0.5))))) {
        coord = xyuv.st * normalDim.st;
        // assume that we have no perspective effect, so use the
        // aperture value from the aperture texture instead of the
        // returned result (xyuv.pq)
        coord0 = floor(coord) + aperture.xy-vec2(0.5,0.5);
        coord1 = rectLinear*coord0+rectOffset;
        coord2 = rectLinear*(coord0 + vec2(0.0,1.0))+rectOffset;
        coord3 = rectLinear*(coord0 + vec2(1.0,1.0))+rectOffset;
        coord4 = rectLinear*(coord0 + vec2(1.0,0.0))+rectOffset;
        fracts = vec4(coord-floor(coord),floor(coord)+vec2(1.0,1.0)-coord);
        color += aperture.z*
            (fracts.z*(fracts.w*texture2D(tex, coord1)+
                       fracts.y*texture2D(tex, coord2)) +
             fracts.x*(fracts.y*texture2D(tex, coord3)+
                       fracts.w*texture2D(tex, coord4)));
      }
    }
    if(aperture.z == 0.0) break;
  }
  gl_FragColor = pow(vec4(  1.0  *gain.rgb*color.rgb+gl_Color.rgb+offset.rgb,gl_Color.a*gain.a*color.a),gamma);
}
