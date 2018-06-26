// Copyright (c) 2013- PPSSPP Project.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2.0 or later versions.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License 2.0 for more details.

// A copy of the GPL 2.0 should have been included with the program.
// If not, see http://www.gnu.org/licenses/

// Official git repository and contact information can be found at
// https://github.com/hrydgard/ppsspp and http://www.ppsspp.org/.

#include <string.h>
#include <algorithm>

#include "profiler/profiler.h"

#include "Common/CPUDetect.h"
#include "Common/MemoryUtil.h"
#include "Core/Config.h"

#include "GPU/Common/SplineCommon.h"
#include "GPU/Common/DrawEngineCommon.h"
#include "GPU/ge_constants.h"
#include "GPU/GPUState.h"  // only needed for UVScale stuff

#if defined(_M_SSE)
#include <emmintrin.h>
#endif


#define START_OPEN 1
#define END_OPEN 2



static void CopyQuad(u8 *&dest, const SimpleVertex *v1, const SimpleVertex *v2, const SimpleVertex *v3, const SimpleVertex *v4) {
	int vertexSize = sizeof(SimpleVertex);
	memcpy(dest, v1, vertexSize);
	dest += vertexSize;
	memcpy(dest, v2, vertexSize);
	dest += vertexSize;
	memcpy(dest, v3, vertexSize);
	dest += vertexSize;
	memcpy(dest, v4, vertexSize);
	dest += vertexSize;
}

static void CopyQuadIndex(u16 *&indices, GEPatchPrimType type, const int idx0, const int idx1, const int idx2, const int idx3) {
	if (type == GE_PATCHPRIM_LINES) {
		*(indices++) = idx0;
		*(indices++) = idx2;
		*(indices++) = idx1;
		*(indices++) = idx3;
		*(indices++) = idx1;
		*(indices++) = idx2;
	}
	else {
		*(indices++) = idx0;
		*(indices++) = idx2;
		*(indices++) = idx1;
		*(indices++) = idx1;
		*(indices++) = idx2;
		*(indices++) = idx3;
	}
}

#undef b2

// Bernstein basis functions
inline float bern0(float x) { return (1 - x) * (1 - x) * (1 - x); }
inline float bern1(float x) { return 3 * x * (1 - x) * (1 - x); }
inline float bern2(float x) { return 3 * x * x * (1 - x); }
inline float bern3(float x) { return x * x * x; }

inline float bern0deriv(float x) { return -3 * (x - 1) * (x - 1); }
inline float bern1deriv(float x) { return 9 * x * x - 12 * x + 3; }
inline float bern2deriv(float x) { return 3 * (2 - 3 * x) * x; }
inline float bern3deriv(float x) { return 3 * x * x; }

// http://en.wikipedia.org/wiki/Bernstein_polynomial
static Math3D::Vec2f Bernstein3D(const Math3D::Vec2f& p0, const Math3D::Vec2f& p1, const Math3D::Vec2f& p2, const Math3D::Vec2f& p3, float x) {
	if (x == 0) return p0;
	else if (x == 1) return p3;
	return p0 * bern0(x) + p1 * bern1(x) + p2 * bern2(x) + p3 * bern3(x);
}

static Vec3f Bernstein3D(const Vec3f& p0, const Vec3f& p1, const Vec3f& p2, const Vec3f& p3, float x) {
	if (x == 0) return p0;
	else if (x == 1) return p3;
	return p0 * bern0(x) + p1 * bern1(x) + p2 * bern2(x) + p3 * bern3(x);
}

static Vec4f Bernstein3D(const Vec4f& p0, const Vec4f& p1, const Vec4f& p2, const Vec4f& p3, float x) {
	if (x == 0) return p0;
	else if (x == 1) return p3;
	return p0 * bern0(x) + p1 * bern1(x) + p2 * bern2(x) + p3 * bern3(x);
}

static Vec4f Bernstein3D(const u32& p0, const u32& p1, const u32& p2, const u32& p3, float x) {
	return Bernstein3D(Vec4f::FromRGBA(p0), Vec4f::FromRGBA(p1), Vec4f::FromRGBA(p2), Vec4f::FromRGBA(p3), x);
}

static Vec3f Bernstein3DDerivative(const Vec3f& p0, const Vec3f& p1, const Vec3f& p2, const Vec3f& p3, float x) {
	return p0 * bern0deriv(x) + p1 * bern1deriv(x) + p2 * bern2deriv(x) + p3 * bern3deriv(x);
}

static void spline_n_4(int i, float t, float *knot, float *splineVal) {
	knot += i + 1;

#ifdef _M_SSE
	const __m128 knot012 = _mm_loadu_ps(&knot[0]);
	const __m128 knot345 = _mm_loadu_ps(&knot[3]);
	const __m128 t012 = _mm_sub_ps(_mm_set_ps1(t), knot012);
	const __m128 f30_41_52 = _mm_div_ps(t012, _mm_sub_ps(knot345, knot012));

	const __m128 knot343 = _mm_shuffle_ps(knot345, knot345, _MM_SHUFFLE(3, 0, 1, 0));
	const __m128 knot122 = _mm_shuffle_ps(knot012, knot012, _MM_SHUFFLE(3, 2, 2, 1));
	const __m128 t122 = _mm_shuffle_ps(t012, t012, _MM_SHUFFLE(3, 2, 2, 1));
	const __m128 f31_42_32 = _mm_div_ps(t122, _mm_sub_ps(knot343, knot122));

	// It's still faster to use SSE, even with this.
	alignas(16) float ff30_41_52[4];
	alignas(16) float ff31_42_32[4];
	_mm_store_ps(ff30_41_52, f30_41_52);
	_mm_store_ps(ff31_42_32, f31_42_32);

	const float &f30 = ff30_41_52[0];
	const float &f41 = ff30_41_52[1];
	const float &f52 = ff30_41_52[2];
	const float &f31 = ff31_42_32[0];
	const float &f42 = ff31_42_32[1];
	const float &f32 = ff31_42_32[2];
#else
	// TODO: Maybe compilers could be coaxed into vectorizing this code without the above explicitly...
	float t0 = (t - knot[0]);
	float t1 = (t - knot[1]);
	float t2 = (t - knot[2]);
	// TODO: All our knots are integers so we should be able to get rid of these divisions (How?)
	float f30 = t0/(knot[3]-knot[0]);
	float f41 = t1/(knot[4]-knot[1]);
	float f52 = t2/(knot[5]-knot[2]);
	float f31 = t1/(knot[3]-knot[1]);
	float f42 = t2/(knot[4]-knot[2]);
	float f32 = t2/(knot[3]-knot[2]);
#endif

	float a = (1-f30)*(1-f31);
	float b = (f31*f41);
	float c = (1-f41)*(1-f42);
	float d = (f42*f52);

	splineVal[0] = a-(a*f32);
	splineVal[1] = 1-a-b+((a+b+c-1)*f32);
	splineVal[2] = b+((1-b-c-d)*f32);
	splineVal[3] = d*f32;
}

// knot should be an array sized n + 5  (n + 1 + 1 + degree (cubic))
static void spline_knot(int n, int type, float *knot) {
	memset(knot, 0, sizeof(float) * (n + 5));
	for (int i = 0; i < n - 1; ++i)
		knot[i + 3] = (float)i;

	if ((type & 1) == 0) {
		knot[0] = -3;
		knot[1] = -2;
		knot[2] = -1;
	}
	if ((type & 2) == 0) {
		knot[n + 2] = (float)(n - 1);
		knot[n + 3] = (float)(n);
		knot[n + 4] = (float)(n + 1);
	} else {
		knot[n + 2] = (float)(n - 2);
		knot[n + 3] = (float)(n - 2);
		knot[n + 4] = (float)(n - 2);
	}
}

// Prepare mesh of one patch for "Instanced Tessellation".
static void TessellateSplinePatchHardware(SplineBezierBatch &spatch) {
	u16 *&indices = spatch.batchIndices;
	int &indexCount = spatch.batchIndexCount;
	int startIndex = spatch.batchVertexCount;

	SimpleVertex *vertices = (SimpleVertex*)spatch.batchVertices;
	float inv_u = 1.0f / (float)spatch.tess_u;
	float inv_v = 1.0f / (float)spatch.tess_v;

	// Generating simple input vertices for the spline-computing vertex shader.
	for (int tile_v = 0; tile_v < spatch.tess_v + 1; ++tile_v) {
		for (int tile_u = 0; tile_u < spatch.tess_u + 1; ++tile_u) {
			SimpleVertex &vert = vertices[tile_v * (spatch.tess_u + 1) + tile_u];
			vert.pos.x = (float)tile_u * inv_u;
			vert.pos.y = (float)tile_v * inv_v;

			// TODO: Move to shader uniform and unify this method spline and bezier if necessary.
			// For compute normal
			vert.nrm.x = inv_u;
			vert.nrm.y = inv_v;
		}
	}

	// Combine the vertices into triangles.
	for (int tile_v = 0; tile_v < spatch.tess_v; ++tile_v) {
		for (int tile_u = 0; tile_u < spatch.tess_u; ++tile_u) {
			int idx0 = tile_v * (spatch.tess_u + 1) + tile_u + startIndex;
			int idx1 = tile_v * (spatch.tess_u + 1) + tile_u + 1 + startIndex;
			int idx2 = (tile_v + 1) * (spatch.tess_u + 1) + tile_u + startIndex;
			int idx3 = (tile_v + 1) * (spatch.tess_u + 1) + tile_u + 1 + startIndex;

			CopyQuadIndex(indices, spatch.primType, idx0, idx1, idx2, idx3);
			indexCount += 6;
		}
	}

	int count = (spatch.tess_v + 1) * (spatch.tess_u + 1);
	spatch.batchVertexCount += count;
	spatch.batchVertices += count * sizeof(SimpleVertex);
}

static void _SplinePatchLowQuality(SplineBezierBatch &spatch) {
	// Fast and easy way - just draw the control points, generate some very basic normal vector substitutes.
	// Very inaccurate but okay for Loco Roco. Maybe should keep it as an option because it's fast.
	u8 *&vertices = spatch.batchVertices;
	u16 *&indices = spatch.batchIndices;
	u32 origVertType = spatch.batchOrigVertType;
	int startIndex = spatch.batchVertexCount;

	const int tile_min_u = (spatch.type_u & START_OPEN) ? 0 : 1;
	const int tile_min_v = (spatch.type_v & START_OPEN) ? 0 : 1;
	const int tile_max_u = (spatch.type_u & END_OPEN) ? spatch.count_u - 1 : spatch.count_u - 2;
	const int tile_max_v = (spatch.type_v & END_OPEN) ? spatch.count_v - 1 : spatch.count_v - 2;

	float tu_width = (float)spatch.count_u - 3.0f;
	float tv_height = (float)spatch.count_v - 3.0f;
	tu_width /= (float)(tile_max_u - tile_min_u);
	tv_height /= (float)(tile_max_v - tile_min_v);

	GEPatchPrimType prim_type = spatch.primType;
	bool computeNormals = spatch.computeNormals;
	bool patchFacing = spatch.patchFacing;

	int i = 0;
	for (int tile_v = tile_min_v; tile_v < tile_max_v; ++tile_v) {
		for (int tile_u = tile_min_u; tile_u < tile_max_u; ++tile_u) {
			int point_index = tile_u + tile_v * spatch.count_u;

			SimpleVertex v0 = *spatch.points[point_index];
			SimpleVertex v1 = *spatch.points[point_index + 1];
			SimpleVertex v2 = *spatch.points[point_index + spatch.count_u];
			SimpleVertex v3 = *spatch.points[point_index + spatch.count_u + 1];

			// Generate UV. TODO: Do this even if UV specified in control points?
			if ((origVertType & GE_VTYPE_TC_MASK) == 0) {
				float u = (tile_u - tile_min_u) * tu_width;
				float v = (tile_v - tile_min_v) * tv_height;

				v0.uv[0] = u;
				v0.uv[1] = v;
				v1.uv[0] = u + tu_width;
				v1.uv[1] = v;
				v2.uv[0] = u;
				v2.uv[1] = v + tv_height;
				v3.uv[0] = u + tu_width;
				v3.uv[1] = v + tv_height;
			}

			// Generate normal if lighting is enabled (otherwise there's no point).
			// This is a really poor quality algorithm, we get facet normals.
			if (computeNormals) {
				Vec3f norm = Cross(v1.pos - v0.pos, v2.pos - v0.pos);
				norm.Normalize();
				if (patchFacing)
					norm *= -1.0f;
				v0.nrm = norm;
				v1.nrm = norm;
				v2.nrm = norm;
				v3.nrm = norm;
			}

			int idx0 = i * 4 + 0 + startIndex;
			int idx1 = i * 4 + 1 + startIndex;
			int idx2 = i * 4 + 2 + startIndex;
			int idx3 = i * 4 + 3 + startIndex;
			i++;

			CopyQuad(vertices, &v0, &v1, &v2, &v3);
			CopyQuadIndex(indices, prim_type, idx0, idx1, idx2, idx3);
			spatch.batchVertexCount += 4;
			spatch.batchIndexCount += 6;
		}
	}
}


static void SplinePatchFullQuality(SplineBezierBatch &spatch, int quality) {
	// Full (mostly) correct tessellation of spline patches.
	// Not very fast.
	u16 *&indices = spatch.batchIndices;
	u32 origVertType = spatch.batchOrigVertType;
	int startIndex = spatch.batchVertexCount;
	int maxVertices = spatch.maxVertexCount;

	// First compute all the vertices and put them in an array
	SimpleVertex *vertices = (SimpleVertex*)spatch.batchVertices;

	bool origNrm = (origVertType & GE_VTYPE_NRM_MASK) != 0;
	bool origCol = (origVertType & GE_VTYPE_COL_MASK) != 0;
	bool origTc = (origVertType & GE_VTYPE_TC_MASK) != 0;

	float *knot_u = new float[spatch.count_u + 4];
	float *knot_v = new float[spatch.count_v + 4];
	spline_knot(spatch.count_u - 1, spatch.type_u, knot_u);
	spline_knot(spatch.count_v - 1, spatch.type_v, knot_v);

	// Increase tessellation based on the size. Should be approximately right?
	int patch_div_s = (spatch.count_u - 3) * spatch.tess_u;
	int patch_div_t = (spatch.count_v - 3) * spatch.tess_v;
	if (quality > 1) {
		// Don't cut below 2, though.
		if (patch_div_s > 2) {
			patch_div_s /= quality;
		}
		if (patch_div_t > 2) {
			patch_div_t /= quality;
		}
	}

	// Downsample until it fits, in case crazy tessellation factors are sent.
	while ((patch_div_s + 1) * (patch_div_t + 1) > maxVertices) {
		patch_div_s /= 2;
		patch_div_t /= 2;
	}

	if (patch_div_s < 1) patch_div_s = 1;
	if (patch_div_t < 1) patch_div_t = 1;

	float tu_width = (float)spatch.count_u - 3.0f;
	float tv_height = (float)spatch.count_v - 3.0f;

	// int max_idx = spatch.count_u * spatch.count_v;

	bool computeNormals = spatch.computeNormals;

	float one_over_patch_div_s = 1.0f / (float)(patch_div_s);
	float one_over_patch_div_t = 1.0f / (float)(patch_div_t);

	for (int tile_v = 0; tile_v < patch_div_t + 1; tile_v++) {
		float v = (float)tile_v * (float)(spatch.count_v - 3) * one_over_patch_div_t;
		if (v < 0.0f)
			v = 0.0f;
		for (int tile_u = 0; tile_u < patch_div_s + 1; tile_u++) {
			float u = (float)tile_u * (float)(spatch.count_u - 3) * one_over_patch_div_s;
			if (u < 0.0f)
				u = 0.0f;
			SimpleVertex *vert = &vertices[tile_v * (patch_div_s + 1) + tile_u];
			Vec4f vert_color(0.0f);
			Vec3f vert_pos;
			vert_pos.SetZero();
			Vec3f vert_nrm;
			if (origNrm) {
				vert_nrm.SetZero();
			}
			if (origCol) {
				vert_color.SetZero();
			} else {
				memcpy(vert->color, spatch.points[0]->color, 4);
			}
			if (origTc) {
				vert->uv[0] = 0.0f;
				vert->uv[1] = 0.0f;
			} else {
				vert->uv[0] = tu_width * ((float)tile_u * one_over_patch_div_s);
				vert->uv[1] = tv_height * ((float)tile_v * one_over_patch_div_t);
			}


			// Collect influences from surrounding control points.
			float u_weights[4];
			float v_weights[4];

			int iu = (int)u;
			int iv = (int)v;

			// TODO: Would really like to fix the surrounding logic somehow to get rid of these but I can't quite get it right..
			// Without the previous epsilons and with large count_u, we will end up doing an out of bounds access later without these.
			if (iu >= spatch.count_u - 3) iu = spatch.count_u - 4;
			if (iv >= spatch.count_v - 3) iv = spatch.count_v - 4;

			spline_n_4(iu, u, knot_u, u_weights);
			spline_n_4(iv, v, knot_v, v_weights);

			// Handle degenerate patches. without this, spatch.points[] may read outside the number of initialized points.
			int patch_w = std::min(spatch.count_u - iu, 4);
			int patch_h = std::min(spatch.count_v - iv, 4);

			for (int ii = 0; ii < patch_w; ++ii) {
				for (int jj = 0; jj < patch_h; ++jj) {
					float u_spline = u_weights[ii];
					float v_spline = v_weights[jj];
					float f = u_spline * v_spline;

					if (f > 0.0f) {
						int idx = spatch.count_u * (iv + jj) + (iu + ii);
						/*
						if (idx >= max_idx) {
							char temp[512];
							snprintf(temp, sizeof(temp), "count_u: %d count_v: %d patch_w: %d patch_h: %d  ii: %d  jj: %d  iu: %d  iv: %d  patch_div_s: %d  patch_div_t: %d\n", spatch.count_u, spatch.count_v, patch_w, patch_h, ii, jj, iu, iv, patch_div_s, patch_div_t);
							OutputDebugStringA(temp);
							Crash();
						}*/

						const SimpleVertex *a = spatch.points[idx];
						// Accumulate Weighted
						vert_pos += Vec3f(a->pos) * f;
						if (origTc) {
							vert->uv[0] += a->uv[0] * f;
							vert->uv[1] += a->uv[1] * f;
						}
						if (origCol) {
							// Accumulate Weighted
							vert_color += Vec4f::FromRGBA(a->color_32) * f;
						}
						if (origNrm) {
							// Accumulate Weighted
							vert_nrm += Vec3f(a->nrm) * f;
						}
					}
				}
			}
			vert->pos = vert_pos;
			if (origNrm) {
				vert_nrm.Normalize();
				vert->nrm = vert_nrm;
			} else {
				vert->nrm.SetZero();
				vert->nrm.z = 1.0f;
			}
			if (origCol) {
				vert->color_32 = vert_color.ToRGBA();
			}
		}
	}

	delete[] knot_u;
	delete[] knot_v;

	// Hacky normal generation through central difference.
	if (computeNormals && !origNrm) {
		for (int v = 0; v < patch_div_t + 1; v++) {
			Vec3f vl_pos = vertices[v * (patch_div_s + 1)].pos;
			Vec3f vc_pos = vertices[v * (patch_div_s + 1)].pos;

			for (int u = 0; u < patch_div_s + 1; u++) {
				const int t = std::max(0, v - 1);
				const int r = std::min(patch_div_s, u + 1);
				const int b = std::min(patch_div_t, v + 1);

				const Vec3f vr_pos = vertices[v * (patch_div_s + 1) + r].pos;

				const Vec3f right = vr_pos - vl_pos;
				const Vec3f down = vertices[b * (patch_div_s + 1) + u].pos - vertices[t * (patch_div_s + 1) + u].pos;

				Vec3f norm = Cross(right, down).Normalized();
				if (spatch.patchFacing) {
					norm *= -1.0f;
				}
				vertices[v * (patch_div_s + 1) + u].nrm = norm;

				// Rotate for the next one to the right.
				vl_pos = vc_pos;
				vc_pos = vr_pos;
			}
		}
	}

	GEPatchPrimType prim_type = spatch.primType;
	// Tessellate.
	for (int tile_v = 0; tile_v < patch_div_t; ++tile_v) {
		for (int tile_u = 0; tile_u < patch_div_s; ++tile_u) {
			int idx0 = tile_v * (patch_div_s + 1) + tile_u + startIndex;
			int idx1 = tile_v * (patch_div_s + 1) + tile_u + 1 + startIndex;
			int idx2 = (tile_v + 1) * (patch_div_s + 1) + tile_u + startIndex;
			int idx3 = (tile_v + 1) * (patch_div_s + 1) + tile_u + 1 + startIndex;

			CopyQuadIndex(indices, prim_type, idx0, idx1, idx2, idx3);
			spatch.batchIndexCount += 6;
		}
	}

	//dest += (indexCount - startIndex) * sizeof(SimpleVertex);
	int count = (patch_div_t + 1) * (patch_div_s + 1);
	spatch.batchVertexCount += count;
	spatch.batchVertices += count * sizeof(SimpleVertex);
}

static void _BezierPatchLowQuality(SplineBezierBatch& bezier) {
	u8 *&dest = bezier.batchVertices;
	u16 *&indices = bezier.batchIndices;
	u32 origVertType = bezier.batchOrigVertType;
	int startIndex = bezier.batchVertexCount;

	const float third = 1.0f / 3.0f;
	// Fast and easy way - just draw the control points, generate some very basic normal vector subsitutes.
	// Very inaccurate though but okay for Loco Roco. Maybe should keep it as an option.

	float u_base = bezier.u_index / 3.0f;
	float v_base = bezier.v_index / 3.0f;

	GEPatchPrimType prim_type = bezier.primType;

	for (int tile_v = 0; tile_v < 3; tile_v++) {
		for (int tile_u = 0; tile_u < 3; tile_u++) {
			int point_index = tile_u + tile_v * 4;

			SimpleVertex v0 = *bezier.points[point_index];
			SimpleVertex v1 = *bezier.points[point_index + 1];
			SimpleVertex v2 = *bezier.points[point_index + 4];
			SimpleVertex v3 = *bezier.points[point_index + 5];

			// Generate UV. TODO: Do this even if UV specified in control points?
			if ((origVertType & GE_VTYPE_TC_MASK) == 0) {
				float u = u_base + tile_u * third;
				float v = v_base + tile_v * third;
				v0.uv[0] = u;
				v0.uv[1] = v;
				v1.uv[0] = u + third;
				v1.uv[1] = v;
				v2.uv[0] = u;
				v2.uv[1] = v + third;
				v3.uv[0] = u + third;
				v3.uv[1] = v + third;
			}

			// Generate normal if lighting is enabled (otherwise there's no point).
			// This is a really poor quality algorithm, we get facet normals.
			if (bezier.computeNormals) {
				Vec3f norm = Cross(v1.pos - v0.pos, v2.pos - v0.pos);
				norm.Normalize();
				if (bezier.patchFacing)
					norm *= -1.0f;
				v0.nrm = norm;
				v1.nrm = norm;
				v2.nrm = norm;
				v3.nrm = norm;
			}

			int total = bezier.index * 3 * 3 * 4; // A patch has 3x3 tiles, and each tiles have 4 vertices.
			int tile_index = tile_u + tile_v * 3;
			int idx0 = total + tile_index * 4 + 0 + startIndex;
			int idx1 = total + tile_index * 4 + 1 + startIndex;
			int idx2 = total + tile_index * 4 + 2 + startIndex;
			int idx3 = total + tile_index * 4 + 3 + startIndex;

			CopyQuad(dest, &v0, &v1, &v2, &v3);
			CopyQuadIndex(indices, prim_type, idx0, idx1, idx2, idx3);
			bezier.batchVertexCount += 4;
			bezier.batchIndexCount += 6;
		}
	}
}

template <typename T>
struct PrecomputedCurves {
	PrecomputedCurves(int count) {
		horiz1 = (T *)AllocateAlignedMemory(count * 4 * sizeof(T), 16);
		horiz2 = horiz1 + count * 1;
		horiz3 = horiz1 + count * 2;
		horiz4 = horiz1 + count * 3;
	}
	~PrecomputedCurves() {
		FreeAlignedMemory(horiz1);
	}

	T Bernstein3D(int u, float bv) {
		return ::Bernstein3D(horiz1[u], horiz2[u], horiz3[u], horiz4[u], bv);
	}

	T Bernstein3DDerivative(int u, float bv) {
		return ::Bernstein3DDerivative(horiz1[u], horiz2[u], horiz3[u], horiz4[u], bv);
	}

	T *horiz1;
	T *horiz2;
	T *horiz3;
	T *horiz4;
};

static void _BezierPatchHighQuality(SplineBezierBatch& bezier, int quality) {
	u16 *&indices = bezier.batchIndices;
	int tess_u = bezier.tess_u / quality;
	int tess_v = bezier.tess_v / quality;
	u32 origVertType = bezier.batchOrigVertType;
	int startIndex = bezier.batchVertexCount;

	const float third = 1.0f / 3.0f;

	// First compute all the vertices and put them in an array
	SimpleVertex *&vertices = (SimpleVertex*&)bezier.batchVertices;

	PrecomputedCurves<Vec3f> prepos(tess_u + 1);
	PrecomputedCurves<Vec4f> precol(tess_u + 1);
	PrecomputedCurves<Math3D::Vec2f> pretex(tess_u + 1);
	PrecomputedCurves<Vec3f> prederivU(tess_u + 1);

	const bool computeNormals = bezier.computeNormals;
	const bool sampleColors = (origVertType & GE_VTYPE_COL_MASK) != 0;
	const bool sampleTexcoords = (origVertType & GE_VTYPE_TC_MASK) != 0;

	// Precompute the horizontal curves to we only have to evaluate the vertical ones.
	for (int i = 0; i < tess_u + 1; i++) {
		float u = ((float)i / (float)tess_u);
		prepos.horiz1[i] = Bernstein3D(bezier.points[0]->pos, bezier.points[1]->pos, bezier.points[2]->pos, bezier.points[3]->pos, u);
		prepos.horiz2[i] = Bernstein3D(bezier.points[4]->pos, bezier.points[5]->pos, bezier.points[6]->pos, bezier.points[7]->pos, u);
		prepos.horiz3[i] = Bernstein3D(bezier.points[8]->pos, bezier.points[9]->pos, bezier.points[10]->pos, bezier.points[11]->pos, u);
		prepos.horiz4[i] = Bernstein3D(bezier.points[12]->pos, bezier.points[13]->pos, bezier.points[14]->pos, bezier.points[15]->pos, u);

		if (sampleColors) {
			precol.horiz1[i] = Bernstein3D(bezier.points[0]->color_32, bezier.points[1]->color_32, bezier.points[2]->color_32, bezier.points[3]->color_32, u);
			precol.horiz2[i] = Bernstein3D(bezier.points[4]->color_32, bezier.points[5]->color_32, bezier.points[6]->color_32, bezier.points[7]->color_32, u);
			precol.horiz3[i] = Bernstein3D(bezier.points[8]->color_32, bezier.points[9]->color_32, bezier.points[10]->color_32, bezier.points[11]->color_32, u);
			precol.horiz4[i] = Bernstein3D(bezier.points[12]->color_32, bezier.points[13]->color_32, bezier.points[14]->color_32, bezier.points[15]->color_32, u);
		}
		if (sampleTexcoords) {
			pretex.horiz1[i] = Bernstein3D(Math3D::Vec2f(bezier.points[0]->uv), Math3D::Vec2f(bezier.points[1]->uv), Math3D::Vec2f(bezier.points[2]->uv), Math3D::Vec2f(bezier.points[3]->uv), u);
			pretex.horiz2[i] = Bernstein3D(Math3D::Vec2f(bezier.points[4]->uv), Math3D::Vec2f(bezier.points[5]->uv), Math3D::Vec2f(bezier.points[6]->uv), Math3D::Vec2f(bezier.points[7]->uv), u);
			pretex.horiz3[i] = Bernstein3D(Math3D::Vec2f(bezier.points[8]->uv), Math3D::Vec2f(bezier.points[9]->uv), Math3D::Vec2f(bezier.points[10]->uv), Math3D::Vec2f(bezier.points[11]->uv), u);
			pretex.horiz4[i] = Bernstein3D(Math3D::Vec2f(bezier.points[12]->uv), Math3D::Vec2f(bezier.points[13]->uv), Math3D::Vec2f(bezier.points[14]->uv), Math3D::Vec2f(bezier.points[15]->uv), u);
		}

		if (computeNormals) {
			prederivU.horiz1[i] = Bernstein3DDerivative(bezier.points[0]->pos, bezier.points[1]->pos, bezier.points[2]->pos, bezier.points[3]->pos, u);
			prederivU.horiz2[i] = Bernstein3DDerivative(bezier.points[4]->pos, bezier.points[5]->pos, bezier.points[6]->pos, bezier.points[7]->pos, u);
			prederivU.horiz3[i] = Bernstein3DDerivative(bezier.points[8]->pos, bezier.points[9]->pos, bezier.points[10]->pos, bezier.points[11]->pos, u);
			prederivU.horiz4[i] = Bernstein3DDerivative(bezier.points[12]->pos, bezier.points[13]->pos, bezier.points[14]->pos, bezier.points[15]->pos, u);
		}
	}


	for (int tile_v = 0; tile_v < tess_v + 1; ++tile_v) {
		for (int tile_u = 0; tile_u < tess_u + 1; ++tile_u) {
			float u = ((float)tile_u / (float)tess_u);
			float v = ((float)tile_v / (float)tess_v);
			float bv = v;

			SimpleVertex &vert = vertices[tile_v * (tess_u + 1) + tile_u];

			if (computeNormals) {
				const Vec3f derivU = prederivU.Bernstein3D(tile_u, bv);
				const Vec3f derivV = prepos.Bernstein3DDerivative(tile_u, bv);
				Vec3f norm = Cross(derivU, derivV).Normalized();
				if (bezier.patchFacing)
					norm *= -1.0f;
				vert.nrm = norm;
			} else {
				vert.nrm.SetZero();
			}

			vert.pos = prepos.Bernstein3D(tile_u, bv);

			if (!sampleTexcoords) {
				// Generate texcoord
				vert.uv[0] = u + bezier.u_index * third;
				vert.uv[1] = v + bezier.v_index * third;
			} else {
				// Sample UV from control points
				const Math3D::Vec2f res = pretex.Bernstein3D(tile_u, bv);
				vert.uv[0] = res.x;
				vert.uv[1] = res.y;
			} 

			if (sampleColors) {
				vert.color_32 = precol.Bernstein3D(tile_u, bv).ToRGBA();
			} else {
				memcpy(vert.color, bezier.points[0]->color, 4);
			}
		}
	}

	GEPatchPrimType prim_type = bezier.primType;
	// Combine the vertices into triangles.
	for (int tile_v = 0; tile_v < tess_v; ++tile_v) {
		for (int tile_u = 0; tile_u < tess_u; ++tile_u) {
			int total = bezier.index * (tess_u + 1) * (tess_v + 1);
			int idx0 = total + tile_v * (tess_u + 1) + tile_u + startIndex;
			int idx1 = total + tile_v * (tess_u + 1) + tile_u + 1 + startIndex;
			int idx2 = total + (tile_v + 1) * (tess_u + 1) + tile_u + startIndex;
			int idx3 = total + (tile_v + 1) * (tess_u + 1) + tile_u + 1 + startIndex;

			CopyQuadIndex(indices, prim_type, idx0, idx1, idx2, idx3);
			bezier.batchIndexCount += 6;
		}
	}
	//dest += (tess_u + 1) * (tess_v + 1) * sizeof(SimpleVertex);
	int count = (tess_u + 1) * (tess_v + 1);
	bezier.batchVertexCount += count;
	bezier.batchVertices += count * sizeof(SimpleVertex);
}

// Prepare mesh of one patch for "Instanced Tessellation".
static void TessellateBezierPatchHardware(SplineBezierBatch& bezier) {
	u16 *&indices = bezier.batchIndices;
	int tess_u = bezier.tess_u;
	int tess_v = bezier.tess_v;
	GEPatchPrimType primType = bezier.primType;

	SimpleVertex *&vertices = (SimpleVertex*&)bezier.batchVertices;

	float inv_u = 1.0f / (float)tess_u;
	float inv_v = 1.0f / (float)tess_v;

	// Generating simple input vertices for the bezier-computing vertex shader.
	for (int tile_v = 0; tile_v < tess_v + 1; ++tile_v) {
		for (int tile_u = 0; tile_u < tess_u + 1; ++tile_u) {
			SimpleVertex &vert = vertices[tile_v * (tess_u + 1) + tile_u];

			vert.pos.x = (float)tile_u * inv_u;
			vert.pos.y = (float)tile_v * inv_v;
		}
	}

	// Combine the vertices into triangles.
	for (int tile_v = 0; tile_v < tess_v; ++tile_v) {
		for (int tile_u = 0; tile_u < tess_u; ++tile_u) {
			int idx0 = tile_v * (tess_u + 1) + tile_u;
			int idx1 = tile_v * (tess_u + 1) + tile_u + 1;
			int idx2 = (tile_v + 1) * (tess_u + 1) + tile_u;
			int idx3 = (tile_v + 1) * (tess_u + 1) + tile_u + 1;

			CopyQuadIndex(indices, primType, idx0, idx1, idx2, idx3);
			bezier.batchIndexCount += 6;
		}
	}

	int count = (tess_u + 1) * (tess_v + 1);
	bezier.batchVertexCount += count;
	bezier.batchVertices += count * sizeof(SimpleVertex);
}

void TessellateBezierPatch(SplineBezierBatch& bezier) {
	switch (g_Config.iSplineBezierQuality) {
	case LOW_QUALITY:
		_BezierPatchLowQuality(bezier);
		break;
	case MEDIUM_QUALITY:
		_BezierPatchHighQuality(bezier, 2);
		break;
	case HIGH_QUALITY:
		_BezierPatchHighQuality(bezier, 1);
		break;
	}
}

void TessellateSplinePatch(SplineBezierBatch &spatch) {
	switch (g_Config.iSplineBezierQuality) {
	case LOW_QUALITY:
		_SplinePatchLowQuality(spatch);
		break;
	case MEDIUM_QUALITY:
		SplinePatchFullQuality(spatch, 2);
		break;
	case HIGH_QUALITY:
		SplinePatchFullQuality(spatch, 1);
		break;
	}
}

// This maps GEPatchPrimType to GEPrimitiveType.
const GEPrimitiveType primType[] = { GE_PRIM_TRIANGLES, GE_PRIM_LINES, GE_PRIM_POINTS, GE_PRIM_POINTS };
static SplineBezierBatch batchLocal{};

void DrawEngineCommon::SubmitSpline(const void *control_points, const void *indices, int tess_u, int tess_v, int count_u, int count_v, int type_u, int type_v, GEPatchPrimType prim_type, bool computeNormals, bool patchFacing, u32 vertType, int *bytesRead) {
	PROFILE_THIS_SCOPE("spline");

	// Real hardware seems to draw nothing when given < 4 either U or V.
	if (count_u < 4 || count_v < 4) {
		return;
	}

	if (batchLocal.batchVertexCount > 0x1FFF) {
		SubmitBatchEnd();
	}

	u16 index_lower_bound = 0;
	u16 index_upper_bound = count_u * count_v - 1;

	// origin
	u32 origVertType = vertType;
	IndexConverter idxConv(origVertType, indices);
	if (indices) {
		GetIndexBounds(indices, count_u * count_v, origVertType, &index_lower_bound, &index_upper_bound);
	}
	VertexDecoder *origVDecoder = GetVertexDecoder((origVertType & 0xFFFFFF) | (gstate.getUVGenMode() << 24));
	*bytesRead = count_u * count_v * origVDecoder->VertexSize();


	// Simplify away bones and morph before proceeding
	SimpleVertex *simplified_control_points = (SimpleVertex *)(decoded + 65536 * 12);
	u8 *temp_buffer = decoded + 65536 * 18;
	vertType = NormalizeVertices((u8 *)simplified_control_points, temp_buffer, (u8 *)control_points, index_lower_bound, index_upper_bound, origVertType);
	VertexDecoder *vdecoder = GetVertexDecoder(vertType);
	int vertexSize = vdecoder->VertexSize();
	if (vertexSize != sizeof(SimpleVertex)) {
		ERROR_LOG(G3D, "Something went really wrong, vertex size: %i vs %i", vertexSize, (int)sizeof(SimpleVertex));
	}

	// TODO: Do something less idiotic to manage this buffer
	auto points = new const SimpleVertex *[count_u * count_v];

	// Make an array of pointers to the control points, to get rid of indices.
	for (int idx = 0; idx < count_u * count_v; idx++) {
		points[idx] = simplified_control_points + (indices ? idxConv.convert(idx) : idx);
	}


	if (batchLocal.batchVertices == nullptr) {
		batchLocal.batchVertices = splineBuffer;
		batchLocal.batchIndices = quadIndices_;
		batchLocal.batchVertexCount = 0;
		batchLocal.batchIndexCount = 0;
		batchLocal.batchPrimType = primType[prim_type];
		batchLocal.batchVertType = vertType;
		batchLocal.batchOrigVertType = origVertType;
	}
	batchLocal.maxVertexCount = SPLINE_BUFFER_SIZE / vertexSize - batchLocal.batchVertexCount;

	//
	batchLocal.tess_u = tess_u;
	batchLocal.tess_v = tess_v;
	batchLocal.type_u = type_u;
	batchLocal.type_v = type_v;
	batchLocal.count_u = count_u;
	batchLocal.count_v = count_v;
	batchLocal.points = points;
	batchLocal.computeNormals = computeNormals;
	batchLocal.primType = prim_type;
	batchLocal.patchFacing = patchFacing;

	if (g_Config.bHardwareTessellation && g_Config.bHardwareTransform && !g_Config.bSoftwareRendering) {
		float *pos = (float*)(decoded + 65536 * 18); // Size 4 float
		float *tex = pos + count_u * count_v * 4; // Size 4 float
		float *col = tex + count_u * count_v * 4; // Size 4 float
		const bool hasColor = (origVertType & GE_VTYPE_COL_MASK) != 0;
		const bool hasTexCoords = (origVertType & GE_VTYPE_TC_MASK) != 0;

		int posStride, texStride, colStride;
		tessDataTransfer->PrepareBuffers(pos, tex, col, posStride, texStride, colStride, count_u * count_v, hasColor, hasTexCoords);
		float *p = pos;
		float *t = tex;
		float *c = col;
		for (int idx = 0; idx < count_u * count_v; idx++) {
			memcpy(p, points[idx]->pos.AsArray(), 3 * sizeof(float));
			p += posStride;
			if (hasTexCoords) {
				memcpy(t, points[idx]->uv, 2 * sizeof(float));
				t += texStride;
			}
			if (hasColor) {
				memcpy(c, Vec4f::FromRGBA(points[idx]->color_32).AsArray(), 4 * sizeof(float));
				c += colStride;
			}
		}
		if (!hasColor)
			memcpy(col, Vec4f::FromRGBA(points[0]->color_32).AsArray(), 4 * sizeof(float));

		tessDataTransfer->SendDataToShader(pos, tex, col, count_u * count_v, hasColor, hasTexCoords);
		TessellateSplinePatchHardware(batchLocal);
		numPatches = (count_u - 3) * (count_v - 3);
	}
	else {
		switch (g_Config.iSplineBezierQuality) {
		case LOW_QUALITY:
			_SplinePatchLowQuality(batchLocal);
			break;
		case MEDIUM_QUALITY:
			SplinePatchFullQuality(batchLocal, 2);
			break;
		case HIGH_QUALITY:
			SplinePatchFullQuality(batchLocal, 1);
			break;
		}
	}
	delete[] points;
}

void DrawEngineCommon::SubmitBezier(const void *control_points, const void *indices, int tess_u, int tess_v, int count_u, int count_v, GEPatchPrimType prim_type, bool computeNormals, bool patchFacing, u32 vertType, int *bytesRead) {
	PROFILE_THIS_SCOPE("bezier");

	u16 index_lower_bound = 0;
	u16 index_upper_bound = count_u * count_v - 1;
	IndexConverter idxConv(vertType, indices);
	if (indices)
		GetIndexBounds(indices, count_u*count_v, vertType, &index_lower_bound, &index_upper_bound);

	VertexDecoder *origVDecoder = GetVertexDecoder((vertType & 0xFFFFFF) | (gstate.getUVGenMode() << 24));
	*bytesRead = count_u * count_v * origVDecoder->VertexSize();

	// Real hardware seems to draw nothing when given < 4 either U or V.
	// This would result in num_patches_u / num_patches_v being 0.
	if (count_u < 4 || count_v < 4) {
		return;
	}

	if (batchLocal.batchVertexCount > 0x1FFF) {
		SubmitBatchEnd();
	}

	// Simplify away bones and morph before proceeding
	// There are normally not a lot of control points so just splitting decoded should be reasonably safe, although not great.
	SimpleVertex *simplified_control_points = (SimpleVertex *)(decoded + 65536 * 12);
	u8 *temp_buffer = decoded + 65536 * 18;
	u32 origVertType = vertType;
	vertType = NormalizeVertices((u8 *)simplified_control_points, temp_buffer, (u8 *)control_points, index_lower_bound, index_upper_bound, vertType);

	VertexDecoder *vdecoder = GetVertexDecoder(vertType);
	int vertexSize = vdecoder->VertexSize();
	if (vertexSize != sizeof(SimpleVertex)) {
		ERROR_LOG(G3D, "Something went really wrong, vertex size: %i vs %i", vertexSize, (int)sizeof(SimpleVertex));
	}

	float *pos = (float*)(decoded + 65536 * 18); // Size 4 float
	float *tex = pos + count_u * count_v * 4; // Size 4 float
	float *col = tex + count_u * count_v * 4; // Size 4 float
	const bool hasColor = (origVertType & GE_VTYPE_COL_MASK) != 0;
	const bool hasTexCoords = (origVertType & GE_VTYPE_TC_MASK) != 0;

	// Bezier patches share less control points than spline patches. Otherwise they are pretty much the same (except bezier don't support the open/close thing)
	int num_patches_u = (count_u - 1) / 3;
	int num_patches_v = (count_v - 1) / 3;

	if (batchLocal.batchVertices == nullptr) {
		batchLocal.batchVertices = splineBuffer;
		batchLocal.batchIndices = quadIndices_;
		batchLocal.batchVertexCount = 0;
		batchLocal.batchIndexCount = 0;
		batchLocal.batchPrimType = primType[prim_type];
		batchLocal.batchVertType = vertType;
		batchLocal.batchOrigVertType = origVertType;
	}
	batchLocal.maxVertexCount = SPLINE_BUFFER_SIZE / vertexSize - batchLocal.batchVertexCount;

	batchLocal.primType = prim_type;
	batchLocal.computeNormals = computeNormals;
	batchLocal.patchFacing = patchFacing;

	// We shouldn't really split up into separate 4x4 patches, instead we should do something that works
	// like the splines, so we subdivide across the whole "mega-patch".

	// If specified as 0, uses 1.
	if (tess_u < 1) {
		tess_u = 1;
	}
	if (tess_v < 1) {
		tess_v = 1;
	}
	batchLocal.tess_u = tess_u;
	batchLocal.tess_v = tess_v;

	if (g_Config.bHardwareTessellation && g_Config.bHardwareTransform && !g_Config.bSoftwareRendering) {
		int posStride, texStride, colStride;
		tessDataTransfer->PrepareBuffers(pos, tex, col, posStride, texStride, colStride, count_u * count_v, hasColor, hasTexCoords);
		float *p = pos;
		float *t = tex;
		float *c = col;
		for (int idx = 0; idx < count_u * count_v; idx++) {
			const SimpleVertex *point = simplified_control_points + (indices ? idxConv.convert(idx) : idx);
			memcpy(p, point->pos.AsArray(), 3 * sizeof(float));
			p += posStride;
			if (hasTexCoords) {
				memcpy(t, point->uv, 2 * sizeof(float));
				t += texStride;
			}
			if (hasColor) {
				memcpy(c, Vec4f::FromRGBA(point->color_32).AsArray(), 4 * sizeof(float));
				c += colStride;
			}
		}
		if (!hasColor) {
			const SimpleVertex *point = simplified_control_points + (indices ? idxConv.convert(0) : 0);
			memcpy(col, Vec4f::FromRGBA(point->color_32).AsArray(), 4 * sizeof(float));
		}

		tessDataTransfer->SendDataToShader(pos, tex, col, count_u * count_v, hasColor, hasTexCoords);
		TessellateBezierPatchHardware(batchLocal);
		numPatches = num_patches_u * num_patches_v;
	} else {
		// Downsample until it fits, in case crazy tessellation factors are sent.
		while ((tess_u + 1) * (tess_v + 1) * num_patches_u * num_patches_v > batchLocal.maxVertexCount) {
			tess_u /= 2;
			tess_v /= 2;
		}
		batchLocal.tess_u = tess_u;
		batchLocal.tess_v = tess_v;


		auto points = new const SimpleVertex *[16];
		batchLocal.points = points;
		for (int patch_u = 0; patch_u < num_patches_u; patch_u++) {
			for (int patch_v = 0; patch_v < num_patches_v; patch_v++) {
				for (int point = 0; point < 16; ++point) {
					int idx = (patch_u * 3 + point % 4) + (patch_v * 3 + point / 4) * count_u;
					points[point] = simplified_control_points + (indices ? idxConv.convert(idx) : idx);
				}
				batchLocal.u_index = patch_u * 3;
				batchLocal.v_index = patch_v * 3;
				batchLocal.index = patch_v * num_patches_u + patch_u;
				TessellateBezierPatch(batchLocal);
			}
		}
		delete[] points;
	}
}


void DrawEngineCommon::SubmitBatchEnd() {
	if (batchLocal.batchVertexCount == 0) {
		DEBUG_LOG(G3D, "SubmitBatchEnd batchLocal.batchVertexCount 0");
		return;
	}

	UVScale prevUVScale;
	if ((batchLocal.batchOrigVertType & GE_VTYPE_TC_MASK) != 0) {
		// We scaled during Normalize already so let's turn it off when drawing.
		prevUVScale = gstate_c.uv;
		gstate_c.uv.uScale = 1.0f;
		gstate_c.uv.vScale = 1.0f;
		gstate_c.uv.uOff = 0.0f;
		gstate_c.uv.vOff = 0.0f;
	}

	u32 vertTypeWithIndex16 = (batchLocal.batchVertType & ~GE_VTYPE_IDX_MASK) | GE_VTYPE_IDX_16BIT;
	uint32_t vertTypeID = GetVertTypeID(vertTypeWithIndex16, gstate.getUVGenMode());

	GEPrimitiveType prim = batchLocal.batchPrimType;
	prevPrim_ = prim;

	// If vtype has changed, setup the vertex decoder.
	if (vertTypeID != lastVType_) {
		dec_ = GetVertexDecoder(vertTypeID);
		lastVType_ = vertTypeID;
	}

	DeferredDrawCall &dc = drawCalls[numDrawCalls++];
	dc.verts = splineBuffer;
	dc.inds = quadIndices_;
	dc.indexType = (vertTypeID & GE_VTYPE_IDX_MASK) >> GE_VTYPE_IDX_SHIFT;
	dc.prim = prim;
	dc.vertexCount = batchLocal.batchIndexCount;
	dc.uvScale = gstate_c.uv;
	dc.cullMode = -1;

	dc.indexLowerBound = 0;
	dc.indexUpperBound = batchLocal.batchVertexCount - 1;
	//GetIndexBounds(dc.inds, splinePatch.batchIndexCount, vertTypeID, &dc.indexLowerBound, &dc.indexUpperBound);

	vertexCountInDrawCalls_ += batchLocal.batchIndexCount;

	if (g_Config.bSoftwareSkinning && (vertTypeID & GE_VTYPE_WEIGHT_MASK)) {
		DecodeVertsStep(decoded, decodeCounter_, decodedVerts_);
		decodeCounter_++;
	}

	DispatchFlush();

	batchLocal.batchVertices = nullptr;
	batchLocal.batchIndices = nullptr;
	batchLocal.batchVertexCount = 0;

	if ((batchLocal.batchOrigVertType & GE_VTYPE_TC_MASK) != 0) {
		gstate_c.uv = prevUVScale;
	}
}
