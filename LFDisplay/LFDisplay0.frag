uniform sampler2D tex;
uniform vec4 gain;
uniform vec4 offset;
uniform vec4 gamma;
    
void main(void) {
  gl_FragColor = texture2D(tex, gl_TexCoord[0].st);
  gl_FragColor = (gain*gl_FragColor+offset+vec4(gl_Color.rgb,1.0))*gl_Color.a;
  gl_FragColor = pow(gl_FragColor, gamma);
}
