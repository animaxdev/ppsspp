
#include "GPU/GLES/GeometryShaderGeneratorGLES.h"
#include "GPU/GLES/ShaderManagerGLES.h"
#include "GPU/Common/ShaderId.h"
#include "gfx_es2/gpu_features.h"

#undef WRITE

#define WRITE p+=sprintf

void GenerateGeometryShader(const VShaderID &id, char *buffer, uint32_t *attrMask, uint64_t *uniformMask) {
	char *p = buffer;

	//
	if (gl_extensions.IsGLES) {
		WRITE(p, "#version 320 es\n");
		WRITE(p, "precision highp float;\n");
	}
	else {
		WRITE(p, "#version 330\n");
	}
	WRITE(p, "layout(lines_adjacency) in;\n");
	WRITE(p, "layout(line_strip, max_vertices=1) out;\n");

	bool doFlatShading = id.Bit(VS_BIT_FLATSHADE);
	bool lmode = id.Bit(VS_BIT_LMODE);
	bool doTexture = id.Bit(VS_BIT_DO_TEXTURE);
	bool enableFog = id.Bit(VS_BIT_ENABLE_FOG);

	const char *shading = doFlatShading ? "flat" : "";


	// in
	WRITE(p, "%s in vec4 v_color0[];\n", shading);
	if (lmode) {
		WRITE(p, "%s in vec3 v_color1[];\n", shading);
	}

	if (doTexture) {
		WRITE(p, "in vec3 v_texcoord[];\n");
	}

	if (enableFog) {
		// See the fragment shader generator
		WRITE(p, "in float v_fogdepth[];\n");
	}


	// out
	WRITE(p, "%s out vec4 g_color0;\n", shading);
	if (lmode) {
		WRITE(p, "%s out vec3 g_color1;\n", shading);
	}

	if (doTexture) {
		WRITE(p, "out vec3 g_texcoord;\n");
	}

	if (enableFog) {
		// See the fragment shader generator
		WRITE(p, "out float g_fogdepth;\n");
	}


	// main
	WRITE(p, "void main() {\n");

	WRITE(p, "    g_color0 = v_color0[0];\n");
	if (lmode) {
		WRITE(p, "    g_color1 = v_color1[0];\n");
	}

	if (doTexture) {
		WRITE(p, "    g_texcoord = v_texcoord[0];\n");
	}

	if (enableFog) {
		// See the fragment shader generator
		WRITE(p, "    g_fogdepth = v_fogdepth[0];\n");
	}

	WRITE(p, "    gl_Position = gl_in[0].gl_Position;\n");
	WRITE(p, "    EmitVertex ();\n");
	WRITE(p, "    EndPrimitive ();\n");

	/*WRITE(p, "    gl_Position = gl_in [0].gl_Position + vec4 (1.0f, 0.0f, 0.0f, 0.0f);\n");
	WRITE(p, "    EmitVertex ();\n");

	WRITE(p, "    gl_Position = gl_in [0].gl_Position + vec4 (1.0f, 1.0f, 0.0f, 0.0f);\n");
	WRITE(p, "    EmitVertex ();\n");

	WRITE(p, "    gl_Position = gl_in [0].gl_Position + vec4 (0.0f, 1.0f, 0.0f, 0.0f);\n");
	WRITE(p, "    EmitVertex ();\n");*/

	WRITE(p, "}\n");


	/*
	WRITE(p, "#version 320 es\n");
	WRITE(p, "layout(lines_adjacency) in;\n");
	WRITE(p, "layout(line_strip, max_vertices=20) out;\n");


	WRITE(p, "void main() {\n");
	WRITE(p, "}\n");
	*/
}
