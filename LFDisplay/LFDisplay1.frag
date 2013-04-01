uniform sampler2D tex;
uniform vec4 gain;
uniform vec4 offset;
uniform vec4 gamma;
uniform mat2 rectLinear;
uniform vec2 rectOffset;
uniform vec4 normalDim;
uniform mat4 RTM;

void main(void) {
  vec2 coord, coord0, coord1, coord2, coord3, coord4;
  vec4 fracts;
  vec4 xyuv;
  vec4 color;
  xyuv = RTM*gl_TexCoord[0];
  if(all(bvec4(lessThanEqual(abs(xyuv.pq),vec2(0.5,0.5)),lessThanEqual(abs(xyuv.st),vec2(0.5,0.5))))) {
    coord = xyuv.st * normalDim.st;
    coord0 = floor(coord) + xyuv.pq;
    coord1 = rectLinear*coord0+rectOffset;
    coord2 = rectLinear*(coord0 + vec2(0.0,1.0))+rectOffset;
    coord3 = rectLinear*(coord0 + vec2(1.0,1.0))+rectOffset;
    coord4 = rectLinear*(coord0 + vec2(1.0,0.0))+rectOffset;
    fracts = vec4(coord-floor(coord),floor(coord)+vec2(1.0,1.0)-coord);
    color = fracts.z*(fracts.w*texture2D(tex, coord1)+
                      fracts.y*texture2D(tex, coord2)) +
            fracts.x*(fracts.y*texture2D(tex, coord3)+
                      fracts.w*texture2D(tex, coord4));
  }
  gl_FragColor = pow(vec4(gain.rgb*color.rgb+gl_Color.rgb+offset.rgb,gl_Color.a*gain.a*color.a),gamma);
}
