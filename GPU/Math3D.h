// Copyright (c) 2012- PPSSPP Project.

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

#pragma once

#include <cmath>

#include "Common/Common.h"
#include "Core/Util/AudioFormat.h"  // for clamp_u8
#include "math/fast/fast_matrix.h"


#if defined(_M_SSE)
#include <emmintrin.h>
#include "Common/CPUDetect.h"

inline __m128 SSECrossProduct(__m128 a, __m128 b)
{
	const __m128 left = _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2)));
	const __m128 right = _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1)));
	return _mm_sub_ps(left, right);
}

inline __m128 SSENormalizeMultiplierSSE2(__m128 v)
{
	const __m128 sq = _mm_mul_ps(v, v);
	const __m128 r2 = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(0, 0, 0, 1));
	const __m128 r3 = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(0, 0, 0, 2));
	const __m128 res = _mm_add_ss(r3, _mm_add_ss(r2, sq));

	const __m128 rt = _mm_rsqrt_ss(res);
	return _mm_shuffle_ps(rt, rt, _MM_SHUFFLE(0, 0, 0, 0));
}

#if _M_SSE >= 0x401
#include <smmintrin.h>

inline __m128 SSENormalizeMultiplierSSE4(__m128 v)
{
	return _mm_rsqrt_ps(_mm_dp_ps(v, v, 0xFF));
}

inline __m128 SSENormalizeMultiplier(__m128 v)
{
	if (cpu_info.bSSE4_1)
		return SSENormalizeMultiplierSSE4(v);
	return SSENormalizeMultiplierSSE2(v);
}
#else
inline __m128 SSENormalizeMultiplier(__m128 v)
{
	return SSENormalizeMultiplierSSE2(v);
}
#endif

#elif PPSSPP_ARCH(ARM_NEON)
#include <arm_neon.h>
#define vectorial_inline    inline
typedef float32x4_t simd4f;
typedef float32x2_t simd2f;

vectorial_inline float simd4f_get_x(simd4f s) { return vgetq_lane_f32(s, 0); }
vectorial_inline float simd4f_get_y(simd4f s) { return vgetq_lane_f32(s, 1); }
vectorial_inline float simd4f_get_z(simd4f s) { return vgetq_lane_f32(s, 2); }
vectorial_inline float simd4f_get_w(simd4f s) { return vgetq_lane_f32(s, 3); }

vectorial_inline simd4f simd4f_splat(float v) {
	simd4f s = vdupq_n_f32(v);
	return s;
}

vectorial_inline simd4f simd4f_splat_x(simd4f v) {
	float32x2_t o = vget_low_f32(v);
	simd4f ret = vdupq_lane_f32(o, 0);
	return ret;
}

vectorial_inline simd4f simd4f_splat_y(simd4f v) {
	float32x2_t o = vget_low_f32(v);
	simd4f ret = vdupq_lane_f32(o, 1);
	return ret;
}

vectorial_inline simd4f simd4f_splat_z(simd4f v) {
	float32x2_t o = vget_high_f32(v);
	simd4f ret = vdupq_lane_f32(o, 0);
	return ret;
}

vectorial_inline simd4f simd4f_splat_w(simd4f v) {
	float32x2_t o = vget_high_f32(v);
	simd4f ret = vdupq_lane_f32(o, 1);
	return ret;
}


vectorial_inline simd4f simd4f_reciprocal(simd4f v) {
	simd4f estimate = vrecpeq_f32(v);
	estimate = vmulq_f32(vrecpsq_f32(estimate, v), estimate);
	estimate = vmulq_f32(vrecpsq_f32(estimate, v), estimate);
	return estimate;
}


vectorial_inline simd4f simd4f_add(simd4f lhs, simd4f rhs) {
	simd4f ret = vaddq_f32(lhs, rhs);
	return ret;
}

vectorial_inline simd4f simd4f_sub(simd4f lhs, simd4f rhs) {
	simd4f ret = vsubq_f32(lhs, rhs);
	return ret;
}

vectorial_inline simd4f simd4f_mul(simd4f lhs, simd4f rhs) {
	simd4f ret = vmulq_f32(lhs, rhs);
	return ret;
}

vectorial_inline simd4f simd4f_div(simd4f lhs, simd4f rhs) {
	simd4f recip = simd4f_reciprocal(rhs);
	simd4f ret = vmulq_f32(lhs, recip);
	return ret;
}


vectorial_inline simd4f simd4f_sum(simd4f v) {
	const simd4f s1 = simd4f_add(simd4f_splat_x(v), simd4f_splat_y(v));
	const simd4f s2 = simd4f_add(s1, simd4f_splat_z(v));
	const simd4f s3 = simd4f_add(s2, simd4f_splat_w(v));
	return s3;
}

vectorial_inline simd4f simd4f_dot2(simd4f lhs, simd4f rhs) {
	const simd4f m = simd4f_mul(lhs, rhs);
	const simd4f s1 = simd4f_add(simd4f_splat_x(m), simd4f_splat_y(m));
	return s1;
}


vectorial_inline float simd4f_dot3_scalar(simd4f lhs, simd4f rhs) {
	const simd4f m = simd4f_mul(lhs, rhs);
	simd2f s1 = vpadd_f32(vget_low_f32(m), vget_low_f32(m));
	s1 = vadd_f32(s1, vget_high_f32(m));
	return vget_lane_f32(s1, 0);
}

vectorial_inline simd4f simd4f_dot3(simd4f lhs, simd4f rhs) {
	return simd4f_splat(simd4f_dot3_scalar(lhs, rhs));
}


vectorial_inline simd4f simd4f_dot4(simd4f lhs, simd4f rhs) {
	return simd4f_sum(simd4f_mul(lhs, rhs));
}

vectorial_inline void simd4f_rsqrt_1iteration(const simd4f& v, simd4f& estimate) {
	simd4f estimate2 = vmulq_f32(estimate, v);
	estimate = vmulq_f32(estimate, vrsqrtsq_f32(estimate2, estimate));
}

vectorial_inline simd4f simd4f_rsqrt(simd4f v) {
	simd4f estimate = vrsqrteq_f32(v);
	simd4f_rsqrt_1iteration(v, estimate);
	simd4f_rsqrt_1iteration(v, estimate);
	return estimate;
}

vectorial_inline simd4f simd4f_sqrt(simd4f v) {
	return vreinterpretq_f32_u32(vandq_u32(vtstq_u32(vreinterpretq_u32_f32(v),
		vreinterpretq_u32_f32(v)),
		vreinterpretq_u32_f32(
			simd4f_reciprocal(simd4f_rsqrt(v)))
	)
	);
}

vectorial_inline simd4f simd4f_length4(simd4f v) {
	return simd4f_sqrt(simd4f_dot4(v, v));
}

vectorial_inline simd4f simd4f_length3(simd4f v) {
	return simd4f_sqrt(simd4f_dot3(v, v));
}

vectorial_inline simd4f simd4f_length2(simd4f v) {
	return simd4f_sqrt(simd4f_dot2(v, v));
}
#endif


namespace Math3D {

// Helper for Vec classes to clamp values.
template<typename T>
inline static T VecClamp(const T &v, const T &low, const T &high)
{
	if (v > high)
		return high;
	if (v < low)
		return low;
	return v;
}

template<typename T>
class Vec2
{
public:
	union
	{
		struct
		{
			T x,y;
		};
#if defined(_M_SSE)
		__m128i ivec;
		__m128 vec;
#elif PPSSPP_ARCH(ARM_NEON)
		int32x4_t ivec;
		float32x4_t vec;
#endif
	};

	T* AsArray() { return &x; }
	const T* AsArray() const { return &x; }

	Vec2() {}
	Vec2(const T a[2]) : x(a[0]), y(a[1]) {}
	Vec2(const T& _x, const T& _y) : x(_x), y(_y) {}
#if defined(_M_SSE)
	Vec2(const __m128 &_vec) : vec(_vec) {}
	Vec2(const __m128i &_ivec) : ivec(_ivec) {}
#elif PPSSPP_ARCH(ARM_NEON)
	Vec2(const float32x4_t &_vec) : vec(_vec) {}
	Vec2(const int32x4_t &_ivec) : ivec(_ivec) {}
#endif

	template<typename T2>
	Vec2<T2> Cast() const
	{
		return Vec2<T2>((T2)x, (T2)y);
	}

	static Vec2 AssignToAll(const T& f)
	{
		return Vec2<T>(f, f);
	}

	void Write(T a[2])
	{
		a[0] = x; a[1] = y;
	}

	Vec2 operator +(const Vec2& other) const
	{
		return Vec2(x+other.x, y+other.y);
	}
	void operator += (const Vec2 &other)
	{
		x+=other.x; y+=other.y;
	}

	Vec2 operator -(const Vec2& other) const
	{
		return Vec2(x-other.x, y-other.y);
	}
	void operator -= (const Vec2& other)
	{
		x-=other.x; y-=other.y;
	}

	Vec2 operator -() const
	{
		return Vec2(-x,-y);
	}

	Vec2 operator * (const Vec2& other) const
	{
		return Vec2(x*other.x, y*other.y);
	}

	
	Vec2 operator * (const T& f) const
	{
		return Vec2(x*f,y*f);
	}
	
	void operator *= (const T& f)
	{
		x*=f; y*=f;
	}

	
	Vec2 operator / (const T& f) const
	{
		return Vec2(x/f,y/f);
	}
	
	void operator /= (const T& f)
	{
		*this = *this / f;
	}

	T Length2() const
	{
		return x*x + y*y;
	}

	Vec2 Clamp(const T &l, const T &h) const
	{
		return Vec2(VecClamp(x, l, h), VecClamp(y, l, h));
	}

	// Only implemented for T=float
	float Length() const;
	void SetLength(const float l);
	float Distance2To(Vec2 &other);
	Vec2 Normalized() const;
	float Normalize(); // returns the previous length, which is often useful

	T& operator [] (int i) //allow vector[1] = 3   (vector.y=3)
	{
		return *((&x) + i);
	}
	T operator [] (const int i) const
	{
		return *((&x) + i);
	}

	void SetZero()
	{
		x=0; y=0;
	}

	// Common aliases: UV (texel coordinates), ST (texture coordinates)
	T& u() { return x; }
	T& v() { return y; }
	T& s() { return x; }
	T& t() { return y; }

	const T& u() const { return x; }
	const T& v() const { return y; }
	const T& s() const { return x; }
	const T& t() const { return y; }
};

typedef Vec2<float> Vec2f;

template<typename T>
class Vec3Packed;


////////////////////////////////////////////////////////////
// Vec3
template<typename T>
class Vec3
{
public:
	union
	{
		struct
		{
			T x,y,z;
		};
#if defined(_M_SSE)
		__m128i ivec;
		__m128 vec;
#elif PPSSPP_ARCH(ARM_NEON)
		int32x4_t ivec;
		float32x4_t vec;
#endif
	};

	T* AsArray() { return &x; }
	const T* AsArray() const { return &x; }

	Vec3() {}
	Vec3(const T a[3]) : x(a[0]), y(a[1]), z(a[2]) {}
	Vec3(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {}
	Vec3(const Vec2<T>& _xy, const T& _z) : x(_xy.x), y(_xy.y), z(_z) {}
	Vec3(const Vec3Packed<T> &_xyz) : x(_xyz.x), y(_xyz.y), z(_xyz.z) {}
#if defined(_M_SSE)
	Vec3(const __m128 &_vec) : vec(_vec) {}
	Vec3(const __m128i &_ivec) : ivec(_ivec) {}
#elif PPSSPP_ARCH(ARM_NEON)
	Vec3(const float32x4_t &_vec) : vec(_vec) {}
	Vec3(const int32x4_t &_ivec) : ivec(_ivec) {}
#endif

	template<typename T2>
	Vec3<T2> Cast() const
	{
		return Vec3<T2>((T2)x, (T2)y, (T2)z);
	}

	// Only implemented for T=int and T=float
	static Vec3 FromRGB(unsigned int rgb);
	unsigned int ToRGB() const; // alpha bits set to zero

	static Vec3 AssignToAll(const T& f)
	{
		return Vec3<T>(f, f, f);
	}

	void Write(T a[3])
	{
		a[0] = x; a[1] = y; a[2] = z;
	}

	Vec3 operator +(const Vec3 &other) const
	{
		return Vec3(x+other.x, y+other.y, z+other.z);
	}
	void operator += (const Vec3 &other)
	{
		x+=other.x; y+=other.y; z+=other.z;
	}

	Vec3 operator -(const Vec3 &other) const
	{
		return Vec3(x-other.x, y-other.y, z-other.z);
	}
	void operator -= (const Vec3 &other)
	{
		x-=other.x; y-=other.y; z-=other.z;
	}

	Vec3 operator -() const
	{
		return Vec3(-x,-y,-z);
	}

	Vec3 operator * (const Vec3 &other) const
	{
		return Vec3(x*other.x, y*other.y, z*other.z);
	}

	
	Vec3 operator * (const T& f) const
	{
		return Vec3(x*f,y*f,z*f);
	}
	
	void operator *= (const T& f)
	{
		x*=f; y*=f; z*=f;
	}


	Vec3 operator / (const T& f) const
	{
		return Vec3(x/f,y/f,z/f);
	}
	
	void operator /= (const T& f)
	{
		*this = *this / f;
	}

	T Length2() const
	{
		return x*x + y*y + z*z;
	}

	Vec3 Clamp(const T &l, const T &h) const
	{
		return Vec3(VecClamp(x, l, h), VecClamp(y, l, h), VecClamp(z, l, h));
	}

	// Only implemented for T=float
	float Length() const;
	void SetLength(const float l);
	float Distance2To(Vec3 &other);
	Vec3 Normalized() const;
	float Normalize(); // returns the previous length, which is often useful

	T& operator [] (int i) //allow vector[2] = 3   (vector.z=3)
	{
		return *((&x) + i);
	}
	T operator [] (const int i) const
	{
		return *((&x) + i);
	}

	void SetZero()
	{
		x=0; y=0; z=0;
	}

	// Common aliases: UVW (texel coordinates), RGB (colors), STQ (texture coordinates)
	T& u() { return x; }
	T& v() { return y; }
	T& w() { return z; }

	T& r() { return x; }
	T& g() { return y; }
	T& b() { return z; }

	T& s() { return x; }
	T& t() { return y; }
	T& q() { return z; }

	const T& u() const { return x; }
	const T& v() const { return y; }
	const T& w() const { return z; }

	const T& r() const { return x; }
	const T& g() const { return y; }
	const T& b() const { return z; }

	const T& s() const { return x; }
	const T& t() const { return y; }
	const T& q() const { return z; }
};


////////////////////////////////////////////////////////////
// Vec3Packed
template<typename T>
class Vec3Packed
{
public:
	T x,y,z;

	Vec3Packed() {}
	Vec3Packed(const T a[3]) : x(a[0]), y(a[1]), z(a[2]) {}
	Vec3Packed(const T& _x, const T& _y, const T& _z) : x(_x), y(_y), z(_z) {}
	Vec3Packed(const Vec2<T>& _xy, const T& _z) : x(_xy.x), y(_xy.y), z(_z) {}
	Vec3Packed(const Vec3<T> &_xyz) : x(_xyz.x), y(_xyz.y), z(_xyz.z) {}

	Vec3<T> operator -(const Vec3Packed &other) const
	{
		return Vec3<T>(x - other.x, y - other.y, z - other.z);
	}

	void SetZero()
	{
		x = 0; y = 0; z = 0;
	}

	const T* AsArray() const { return &x; }
};


////////////////////////////////////////////////////////////
// Vec4
template<typename T>
class Vec4
{
public:
	union
	{
		struct
		{
			T x,y,z,w;
		};
#if defined(_M_SSE)
		__m128i ivec;
		__m128 vec;
#elif PPSSPP_ARCH(ARM_NEON)
		int32x4_t ivec;
		float32x4_t vec;
#endif
	};

	T* AsArray() { return &x; }
	const T* AsArray() const { return &x; }

	Vec4() {}
	Vec4(const T v) : x(v), y(v), z(v), w(v) {}
	Vec4(const T a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}
	Vec4(const T& _x, const T& _y, const T& _z, const T& _w) : x(_x), y(_y), z(_z), w(_w) {}
	Vec4(const Vec2<T>& _xy, const T& _z, const T& _w) : x(_xy.x), y(_xy.y), z(_z), w(_w) {}
	Vec4(const Vec3<T>& _xyz, const T& _w) : x(_xyz.x), y(_xyz.y), z(_xyz.z), w(_w) {}
#if defined(_M_SSE)
	Vec4(const __m128 &_vec) : vec(_vec) {}
	Vec4(const __m128i &_ivec) : ivec(_ivec) {}
#elif PPSSPP_ARCH(ARM_NEON)
	Vec4(const float32x4_t &_vec) : vec(_vec) {}
	Vec4(const int32x4_t &_ivec) : ivec(_ivec) {}
#endif

	template<typename T2>
	Vec4<T2> Cast() const
	{
		return Vec4<T2>((T2)x, (T2)y, (T2)z, (T2)w);
	}

	// Only implemented for T=int and T=float
	static Vec4 FromRGBA(unsigned int rgba);
	unsigned int ToRGBA() const;
	void ToRGBA(u8 *rgba) const;

	static Vec4 AssignToAll(const T& f)
	{
		return Vec4<T>(f, f, f, f);
	}

	void Write(T a[4])
	{
		a[0] = x; a[1] = y; a[2] = z; a[3] = w;
	}

	Vec4 operator +(const Vec4& other) const
	{
		return Vec4(x+other.x, y+other.y, z+other.z, w+other.w);
	}
	void operator += (const Vec4& other)
	{
		x+=other.x; y+=other.y; z+=other.z; w+=other.w;
	}
	Vec4 operator -(const Vec4 &other) const
	{
		return Vec4(x-other.x, y-other.y, z-other.z, w-other.w);
	}
	void operator -= (const Vec4 &other)
	{
		x-=other.x; y-=other.y; z-=other.z; w-=other.w;
	}
	Vec4 operator -() const
	{
		return Vec4(-x,-y,-z,-w);
	}
	Vec4 operator * (const Vec4 &other) const
	{
		return Vec4(x*other.x, y*other.y, z*other.z, w*other.w);
	}
	Vec4 operator | (const Vec4 &other) const
	{
		return Vec4(x | other.x, y | other.y, z | other.z, w | other.w);
	}
	
	Vec4 operator * (const T& f) const
	{
		return Vec4(x*f,y*f,z*f,w*f);
	}
	
	void operator *= (const T& f)
	{
		x*=f; y*=f; z*=f; w*=f;
	}

	Vec4 operator / (const T& f) const
	{
		return Vec4(x/f,y/f,z/f,w/f);
	}
	
	void operator /= (const T& f)
	{
		*this = *this / f;
	}

	T Length2() const
	{
		return x*x + y*y + z*z + w*w;
	}

	Vec4 Clamp(const T &l, const T &h) const
	{
		return Vec4(VecClamp(x, l, h), VecClamp(y, l, h), VecClamp(z, l, h), VecClamp(w, l, h));
	}

	Vec4 Reciprocal() const
	{
		const T one = 1.0f;
		return Vec4(one / x, one / y, one / z, one / w);
	}

	// Only implemented for T=float
	float Length() const;
	void SetLength(const float l);
	float Distance2To(Vec4 &other);
	Vec4 Normalized() const;
	float Normalize(); // returns the previous length, which is often useful

	T& operator [] (int i) //allow vector[2] = 3   (vector.z=3)
	{
		return *((&x) + i);
	}
	T operator [] (const int i) const
	{
		return *((&x) + i);
	}

	void SetZero()
	{
		x=0; y=0; z=0; w=0;
	}

	// Common alias: RGBA (colors)
	T& r() { return x; }
	T& g() { return y; }
	T& b() { return z; }
	T& a() { return w; }

	const T& r() const { return x; }
	const T& g() const { return y; }
	const T& b() const { return z; }
	const T& a() const { return w; }

	const Vec3<T> rgb() const { return Vec3<T>(x, y, z); }
};


////////////////////////////////////////////////////////////
// Mat3x3
template<typename BaseType>
class Mat3x3
{
public:
	// Convention: first three values = first column
	Mat3x3(const BaseType values[])
	{
		for (unsigned int i = 0; i < 3*3; ++i)
		{
			this->values[i] = values[i];
		}
	}

	Mat3x3(BaseType _00, BaseType _01, BaseType _02, BaseType _10, BaseType _11, BaseType _12, BaseType _20, BaseType _21, BaseType _22)
	{
		values[0] = _00;
		values[1] = _01;
		values[2] = _02;
		values[3] = _10;
		values[4] = _11;
		values[5] = _12;
		values[6] = _20;
		values[7] = _21;
		values[8] = _22;
	}

	template<typename T>
	Vec3<T> operator * (const Vec3<T>& vec) const
	{
		Vec3<T> ret;
		ret.x = values[0]*vec.x + values[3]*vec.y + values[6]*vec.z;
		ret.y = values[1]*vec.x + values[4]*vec.y + values[7]*vec.z;
		ret.z = values[2]*vec.x + values[5]*vec.y + values[8]*vec.z;
		return ret;
	}

	Mat3x3 Inverse() const
	{
		float a = values[0];
		float b = values[1];
		float c = values[2];
		float d = values[3];
		float e = values[4];
		float f = values[5];
		float g = values[6];
		float h = values[7];
		float i = values[8];
		return Mat3x3(e*i-f*h, f*g-d*i, d*h-e*g,
						c*h-b*i, a*i-c*g, b*g-a*h,
						b*f-c*e, c*d-a*f, a*e-b*d) / Det();
	}

	BaseType Det() const
	{
		return values[0]*values[4]*values[8] + values[3]*values[7]*values[2] +
				values[6]*values[1]*values[5] - values[2]*values[4]*values[6] -
				values[5]*values[7]*values[0] - values[8]*values[1]*values[3];
	}

	Mat3x3 operator / (const BaseType& val) const
	{
		return Mat3x3(values[0]/val, values[1]/val, values[2]/val,
						values[3]/val, values[4]/val, values[5]/val,
						values[6]/val, values[7]/val, values[8]/val);
	}

private:
	BaseType values[3*3];
};


////////////////////////////////////////////////////////////
// Mat4x4
template<typename BaseType>
class Mat4x4
{
public:
	// Convention: first four values in arrow = first column
	Mat4x4(const BaseType values[])
	{
		for (unsigned int i = 0; i < 4*4; ++i)
		{
			this->values[i] = values[i];
		}
	}

	template<typename T>
	Vec4<T> operator * (const Vec4<T>& vec) const
	{
		Vec4<T> ret;
		ret.x = values[0]*vec.x + values[4]*vec.y + values[8]*vec.z + values[12]*vec.w;
		ret.y = values[1]*vec.x + values[5]*vec.y + values[9]*vec.z + values[13]*vec.w;
		ret.z = values[2]*vec.x + values[6]*vec.y + values[10]*vec.z + values[14]*vec.w;
		ret.w = values[3]*vec.x + values[7]*vec.y + values[11]*vec.z + values[15]*vec.w;
		return ret;
	}

private:
	BaseType values[4*4];
};



template<typename T>
inline T Dot(const Vec2<T>& a, const Vec2<T>& b)
{
	return a.x*b.x + a.y*b.y;
}

template<typename T>
inline T Dot(const Vec3<T>& a, const Vec3<T>& b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

template<typename T>
inline T Dot(const Vec4<T>& a, const Vec4<T>& b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

template<typename T>
inline Vec3<T> Cross(const Vec3<T>& a, const Vec3<T>& b)
{
	return Vec3<T>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

template<>
inline Vec3<float> Cross(const Vec3<float>& a, const Vec3<float>& b)
{
#if defined(_M_SSE)
	return SSECrossProduct(a.vec, b.vec);
#else
	return Vec3<float>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
#endif
}

////////////////////////////////////////////////////////////
// Vec4
template<>
inline Vec3<float> Vec3<float>::FromRGB(unsigned int rgb)
{
#if defined(_M_SSE)
	__m128i z = _mm_setzero_si128();
	__m128i c = _mm_cvtsi32_si128(rgb);
	c = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c, z), z);
	return Vec3<float>(_mm_mul_ps(_mm_cvtepi32_ps(c), _mm_set_ps1(1.0f / 255.0f)));
#else
	return Vec3((rgb & 0xFF) * (1.0f / 255.0f),
		((rgb >> 8) & 0xFF) * (1.0f / 255.0f),
		((rgb >> 16) & 0xFF) * (1.0f / 255.0f));
#endif
}

template<>
inline Vec3<int> Vec3<int>::FromRGB(unsigned int rgb)
{
#if defined(_M_SSE)
	__m128i z = _mm_setzero_si128();
	__m128i c = _mm_cvtsi32_si128(rgb);
	c = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c, z), z);
	return Vec3<int>(c);
#else
	return Vec3(rgb & 0xFF, (rgb >> 8) & 0xFF, (rgb >> 16) & 0xFF);
#endif
}

template<>
__forceinline unsigned int Vec3<float>::ToRGB() const
{
#if defined(_M_SSE)
	__m128i c = _mm_cvtps_epi32(_mm_mul_ps(vec, _mm_set_ps1(255.0f)));
	__m128i c16 = _mm_packs_epi32(c, c);
	return _mm_cvtsi128_si32(_mm_packus_epi16(c16, c16)) & 0x00FFFFFF;
#else
	return (clamp_u8((int)(r() * 255.f)) << 0) |
		(clamp_u8((int)(g() * 255.f)) << 8) |
		(clamp_u8((int)(b() * 255.f)) << 16);
#endif
}

template<>
__forceinline unsigned int Vec3<int>::ToRGB() const
{
#if defined(_M_SSE)
	__m128i c16 = _mm_packs_epi32(ivec, ivec);
	return _mm_cvtsi128_si32(_mm_packus_epi16(c16, c16)) & 0x00FFFFFF;
#else
	return clamp_u8(r()) | (clamp_u8(g()) << 8) | (clamp_u8(b()) << 16);
#endif
}



////////////////////////////////////////////////////////////
// Vec4
template<>
inline Vec4<float> Vec4<float>::FromRGBA(unsigned int rgba)
{
#if defined(_M_SSE)
	__m128i z = _mm_setzero_si128();
	__m128i c = _mm_cvtsi32_si128(rgba);
	c = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c, z), z);
	return Vec4<float>(_mm_mul_ps(_mm_cvtepi32_ps(c), _mm_set_ps1(1.0f / 255.0f)));
#else
	return Vec4((rgba & 0xFF) * (1.0f / 255.0f),
		((rgba >> 8) & 0xFF) * (1.0f / 255.0f),
		((rgba >> 16) & 0xFF) * (1.0f / 255.0f),
		((rgba >> 24) & 0xFF) * (1.0f / 255.0f));
#endif
}

template<>
inline Vec4<int> Vec4<int>::FromRGBA(unsigned int rgba)
{
#if defined(_M_SSE)
	__m128i z = _mm_setzero_si128();
	__m128i c = _mm_cvtsi32_si128(rgba);
	c = _mm_unpacklo_epi16(_mm_unpacklo_epi8(c, z), z);
	return Vec4<int>(c);
#else
	return Vec4(rgba & 0xFF, (rgba >> 8) & 0xFF, (rgba >> 16) & 0xFF, (rgba >> 24) & 0xFF);
#endif
}

template<>
__forceinline unsigned int Vec4<float>::ToRGBA() const
{
#if defined(_M_SSE)
	__m128i c = _mm_cvtps_epi32(_mm_mul_ps(vec, _mm_set_ps1(255.0f)));
	__m128i c16 = _mm_packs_epi32(c, c);
	return _mm_cvtsi128_si32(_mm_packus_epi16(c16, c16));
#else
	return (clamp_u8((int)(r() * 255.f)) << 0) |
		(clamp_u8((int)(g() * 255.f)) << 8) |
		(clamp_u8((int)(b() * 255.f)) << 16) |
		(clamp_u8((int)(a() * 255.f)) << 24);
#endif
}

template<>
__forceinline unsigned int Vec4<int>::ToRGBA() const
{
#if defined(_M_SSE)
	__m128i c16 = _mm_packs_epi32(ivec, ivec);
	return _mm_cvtsi128_si32(_mm_packus_epi16(c16, c16));
#else
	return clamp_u8(r()) | (clamp_u8(g()) << 8) | (clamp_u8(b()) << 16) | (clamp_u8(a()) << 24);
#endif
}

template<typename T>
__forceinline void Vec4<T>::ToRGBA(u8 *rgba) const
{
	*(u32 *)rgba = ToRGBA();
}


//##############################################################
////////////////////////////////////////////////////////////////
// Vec2

template<>
inline float Vec2<float>::Length() const
{
#if defined(_M_SSE)
	float ret;
	__m128 xy = _mm_loadu_ps(&x);
	__m128 sq = _mm_mul_ps(xy, xy);
	const __m128 r2 = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(0, 0, 0, 1));
	const __m128 res = _mm_add_ss(sq, r2);
	_mm_store_ss(&ret, _mm_sqrt_ss(res));
	return ret;
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_get_x(simd4f_length2(vec));
#else
	return sqrtf(Length2());
#endif
}

template<>
inline void Vec2<float>::SetLength(const float l)
{
	(*this) *= l / Length();
}

template<>
inline float Vec2<float>::Distance2To(Vec2<float> &other)
{
	return Vec2<float>(other - (*this)).Length2();
}

template<>
inline Vec2<float> Vec2<float>::Normalized() const
{
	return (*this) / Length();
}

template<>
inline float Vec2<float>::Normalize()
{
	float len = Length();
	(*this) = (*this) / len;
	return len;
}


////////////////////////////////////////////////////////////////
// Vec3

template<>
inline Vec3<float> Vec3<float>::operator +(const Vec3 &other) const
{
#if defined(_M_SSE)
	return _mm_add_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_add(vec, other.vec);
#else
	return Vec3(x + other.x, y + other.y, z + other.z);
#endif
}

template<>
inline void Vec3<float>::operator +=(const Vec3 &other)
{
#if defined(_M_SSE)
	vec = _mm_add_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	vec = simd4f_add(vec, other.vec);
#else
	x += other.x; y += other.y; z += other.z;
#endif
}


template<>
inline Vec3<float> Vec3<float>::operator -(const Vec3 &other) const
{
#if defined(_M_SSE)
	return _mm_sub_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_sub(vec, other.vec);
#else
	return Vec3(x - other.x, y - other.y, z - other.z);
#endif
}

template<>
inline void Vec3<float>::operator -=(const Vec3 &other)
{
#if defined(_M_SSE)
	vec = _mm_sub_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	vec = simd4f_sub(vec, other.vec);
#else
	x -= other.x; y -= other.y; z -= other.z;
#endif
}


template<>
inline Vec3<float> Vec3<float>::operator * (const Vec3 &other) const
{
#if defined(_M_SSE)
	return _mm_mul_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_mul(vec, other.vec);
#else
	return Vec3(x*other.x, y*other.y, z*other.z);
#endif
}

template<>
inline Vec3<float> Vec3<float>::operator * (const float& f) const
{
#if defined(_M_SSE)
	__m128 a = _mm_set_ps1(f);
	return _mm_mul_ps(vec, a);
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_mul(vec, simd4f_splat(f));
#else
	return Vec3(x*f, y*f, z*f);
#endif
}

template<>
inline void Vec3<float>::operator *= (const float& f)
{
#if defined(_M_SSE)
	__m128 a = _mm_set_ps1(f);
	vec = _mm_mul_ps(vec, a);
#elif PPSSPP_ARCH(ARM_NEON)
	vec = simd4f_mul(vec, simd4f_splat(f));
#else
	x *= f; y *= f; z *= f;
#endif
}

template<>
inline Vec3<float> Vec3<float>::operator / (const float& f) const
{
#if defined(_M_SSE)
	return _mm_div_ps(vec, _mm_set_ps1(f));
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_div(vec, simd4f_splat(f));
#else
	return Vec3(x / f, y / f, z / f, w / f);
#endif
}

template<>
inline float Vec3<float>::Length() const
{
#if defined(_M_SSE)
	float ret;
	__m128 xyz = _mm_loadu_ps(&x);
	__m128 sq = _mm_mul_ps(xyz, xyz);
	const __m128 r2 = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(0, 0, 0, 1));
	const __m128 r3 = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(0, 0, 0, 2));
	const __m128 res = _mm_add_ss(sq, _mm_add_ss(r2, r3));
	_mm_store_ss(&ret, _mm_sqrt_ss(res));
	return ret;
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_get_x(simd4f_length3(vec));
#else
	return sqrtf(Length2());
#endif
}

template<>
inline void Vec3<float>::SetLength(const float l)
{
	(*this) *= l / Length();
}



template<>
inline float Vec3<float>::Distance2To(Vec3<float> &other)
{
	return Vec3<float>(other - (*this)).Length2();
}

template<>
inline Vec3<float> Vec3<float>::Normalized() const
{
#if defined(_M_SSE)
	__m128 normalize = SSENormalizeMultiplier(vec);
	return _mm_mul_ps(vec, normalize);
#elif PPSSPP_ARCH(ARM_NEON)
	simd4f invlen = simd4f_rsqrt(simd4f_dot4(vec, vec));
	return simd4f_mul(vec, invlen);
#else
	return (*this) / Length();
#endif
}

template<>
inline float Vec3<float>::Normalize()
{
	float len = Length();
	(*this) = (*this) / len;
	return len;
}


////////////////////////////////////////////////////////////////
// Vec4

template<>
inline Vec4<float> Vec4<float>::operator +(const Vec4 &other) const
{
#if defined(_M_SSE)
	return _mm_add_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_add(vec, other.vec);
#else
	return Vec4(x + other.x, y + other.y, z + other.z, w + other.w);
#endif
}

template<>
inline void Vec4<float>::operator +=(const Vec4 &other)
{
#if defined(_M_SSE)
	vec = _mm_add_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	vec = simd4f_add(vec, other.vec);
#else
	x += other.x; y += other.y; z += other.z; w += other.w;
#endif
}


template<>
inline Vec4<float> Vec4<float>::operator -(const Vec4 &other) const
{
#if defined(_M_SSE)
	return _mm_sub_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_sub(vec, other.vec);
#else
	return Vec4(x - other.x, y - other.y, z - other.z, w - other.w);
#endif
}

template<>
inline void Vec4<float>::operator -=(const Vec4 &other)
{
#if defined(_M_SSE)
	vec = _mm_sub_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	vec = simd4f_sub(vec, other.vec);
#else
	x -= other.x; y -= other.y; z -= other.z; w -= other.w;
#endif
}


template<>
inline Vec4<float> Vec4<float>::operator * (const Vec4 &other) const
{
#if defined(_M_SSE)
	return _mm_mul_ps(vec, other.vec);
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_mul(vec, other.vec);
#else
	return Vec4(x*other.x, y*other.y, z*other.z, w*other.w);
#endif
}

template<>
inline Vec4<float> Vec4<float>::operator / (const float& f) const
{
#if defined(_M_SSE)
	return _mm_div_ps(vec, _mm_set_ps1(f));
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_div(vec, simd4f_splat(f));
#else
	return Vec4(x / f, y / f, z / f, w / f);
#endif
}


#if defined(_M_SSE) || defined(HAVE_NEON)
template<>
inline Vec4<float>::Vec4(const float v)
#if defined(_M_SSE)
	: vec(_mm_set_ps1(v))
#elif PPSSPP_ARCH(ARM_NEON)
	: vec(simd4f_splat(v))
#endif
{}
#endif

template<>
inline float Vec4<float>::Length() const
{
#if defined(_M_SSE)
	float ret;
	__m128 xyzw = _mm_loadu_ps(&x);
	__m128 sq = _mm_mul_ps(xyzw, xyzw);
	const __m128 r2 = _mm_add_ps(sq, _mm_movehl_ps(sq, sq));
	const __m128 res = _mm_add_ss(r2, _mm_shuffle_ps(r2, r2, _MM_SHUFFLE(0, 0, 0, 1)));
	_mm_store_ss(&ret, _mm_sqrt_ss(res));
	return ret;
#elif PPSSPP_ARCH(ARM_NEON)
	return simd4f_get_x(simd4f_length4(vec));
#else
	return sqrtf(Length2());
#endif
}

template<>
inline void Vec4<float>::SetLength(const float l)
{
	(*this) *= l / Length();
}

template<>
inline float Vec4<float>::Distance2To(Vec4<float> &other)
{
	return Vec4<float>(other - (*this)).Length2();
}

template<>
inline Vec4<float> Vec4<float>::Normalized() const
{
#if defined(_M_SSE)
	__m128 normalize = SSENormalizeMultiplier(vec);
	return _mm_mul_ps(vec, normalize);
#elif PPSSPP_ARCH(ARM_NEON)
	simd4f invlen = simd4f_rsqrt(simd4f_dot4(vec, vec));
	return simd4f_mul(vec, invlen);
#else
	return (*this) / Length();
#endif
}

template<>
inline float Vec4<float>::Normalize()
{
	float len = Length();
	(*this) = (*this) / len;
	return len;
}



}; // namespace Math3D



////////////////////////////////////////////////////////////
// extra

typedef Math3D::Vec3<float> Vec3f;
typedef Math3D::Vec3Packed<float> Vec3Packedf;
typedef Math3D::Vec4<float> Vec4f;

// v and vecOut must point to different memory.
inline void Vec3ByMatrix43(float vecOut[3], const float v[3], const float m[12]) {
	vecOut[0] = v[0] * m[0] + v[1] * m[3] + v[2] * m[6] + m[9];
	vecOut[1] = v[0] * m[1] + v[1] * m[4] + v[2] * m[7] + m[10];
	vecOut[2] = v[0] * m[2] + v[1] * m[5] + v[2] * m[8] + m[11];
}

inline void Vec3ByMatrix44(float vecOut[4], const float v[3], const float m[16])
{
	vecOut[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + m[12];
	vecOut[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + m[13];
	vecOut[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + m[14];
	vecOut[3] = v[0] * m[3] + v[1] * m[7] + v[2] * m[11] + m[15];
}

/*inline void Vec4ByMatrix44(float vecOut[4], const float v[4], const float m[16])
{
	vecOut[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + v[3] * m[12];
	vecOut[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + v[3] * m[13];
	vecOut[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + v[3] * m[14];
	vecOut[3] = v[0] * m[3] + v[1] * m[7] + v[2] * m[11] + v[3] * m[15];
}*/

inline void Norm3ByMatrix43(float vecOut[3], const float v[3], const float m[12])
{
	vecOut[0] = v[0] * m[0] + v[1] * m[3] + v[2] * m[6];
	vecOut[1] = v[0] * m[1] + v[1] * m[4] + v[2] * m[7];
	vecOut[2] = v[0] * m[2] + v[1] * m[5] + v[2] * m[8];
}

inline void Matrix4ByMatrix4(float out[16], const float a[16], const float b[16]) {
	fast_matrix_mul_4x4(out, b, a);
}

inline void ConvertMatrix4x3To4x4(float *m4x4, const float *m4x3) {
	m4x4[0] = m4x3[0];
	m4x4[1] = m4x3[1];
	m4x4[2] = m4x3[2];
	m4x4[3] = 0.0f;
	m4x4[4] = m4x3[3];
	m4x4[5] = m4x3[4];
	m4x4[6] = m4x3[5];
	m4x4[7] = 0.0f;
	m4x4[8] = m4x3[6];
	m4x4[9] = m4x3[7];
	m4x4[10] = m4x3[8];
	m4x4[11] = 0.0f;
	m4x4[12] = m4x3[9];
	m4x4[13] = m4x3[10];
	m4x4[14] = m4x3[11];
	m4x4[15] = 1.0f;
}

inline void ConvertMatrix4x3To4x4Transposed(float *m4x4, const float *m4x3) {
	m4x4[0] = m4x3[0];
	m4x4[1] = m4x3[3];
	m4x4[2] = m4x3[6];
	m4x4[3] = m4x3[9];
	m4x4[4] = m4x3[1];
	m4x4[5] = m4x3[4];
	m4x4[6] = m4x3[7];
	m4x4[7] = m4x3[10];
	m4x4[8] = m4x3[2];
	m4x4[9] = m4x3[5];
	m4x4[10] = m4x3[8];
	m4x4[11] = m4x3[11];
	m4x4[12] = 0.0f;
	m4x4[13] = 0.0f;
	m4x4[14] = 0.0f;
	m4x4[15] = 1.0f;
}

// 0369
// 147A
// 258B
// ->>-
// 0123
// 4567
// 89AB
// Don't see a way to SIMD that. Should be pretty fast anyway.
inline void ConvertMatrix4x3To3x4Transposed(float *m4x4, const float *m4x3) {
	m4x4[0] = m4x3[0];
	m4x4[1] = m4x3[3];
	m4x4[2] = m4x3[6];
	m4x4[3] = m4x3[9];
	m4x4[4] = m4x3[1];
	m4x4[5] = m4x3[4];
	m4x4[6] = m4x3[7];
	m4x4[7] = m4x3[10];
	m4x4[8] = m4x3[2];
	m4x4[9] = m4x3[5];
	m4x4[10] = m4x3[8];
	m4x4[11] = m4x3[11];
}

inline void Transpose4x4(float out[16], const float in[16]) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			out[i * 4 + j] = in[j * 4 + i];
		}
	}
}


// linear interpolation via float: 0.0=begin, 1.0=end
template<typename X>
inline X Lerp(const X& begin, const X& end, const float t)
{
	return begin*(1.f-t) + end*t;
}

// linear interpolation via int: 0=begin, base=end
template<typename X, int base>
inline X LerpInt(const X& begin, const X& end, const int t)
{
	return (begin*(base-t) + end*t) / base;
}
