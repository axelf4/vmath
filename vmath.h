/**
 * Vector math library with SIMD acceleration.
 * @file vmath.h
 */

#ifndef VMATH_H
#define VMATH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <string.h>

/**
 * Forces data to be n-byte aligned.
 * @def ALIGN(n)
 * @param n The byte boundary to align to.
 *
 * SSE requires data to be aligned on 16 bytes boundary.
 */
#if __cplusplus >= 201103L || (defined __DOXYGEN__)
#define ALIGN(n) alignas(n)
#elif (defined __GNUC__) || (defined __PGI) || (defined __IBMCPP__) || (defined __ARMCC_VERSION)
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined _MSC_VER
#define ALIGN(n) __declspec(align(n))
#else
#error Do not know the equivalent of alignas(n) for your compiler
#endif

/** Inlining. */
#define VMATH_INLINE static inline

#ifdef __DOXYGEN__
/** If defined, compiled code won't be SIMD accelerated even if possible. */
#define VMATH_NO_INTRINSICS
#endif

#ifndef VMATH_NO_INTRINSICS
#if defined(_MSC_VER)
#include <intrin.h> // Microsoft C/C++-compatible compiler
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// #include <x86intrin.h> // GCC-compatible compiler, targeting x86/x86-64
#include <immintrin.h>
#endif

#define VMATH_SSE_INTRINSICS
#define VMATH_SSE2_INTRINSICS
#ifdef __SSE3__
#define VMATH_SSE3_INTRINSICS
#endif
#ifdef __SSE4_1__
#define VMATH_SSE4_1_INTRINSICS
#endif
#ifdef __SSE4_2__
#define VMATH_SSE4_2_INTRINSICS
#endif

#ifndef _MM_SHUFFLE
#define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#endif
#endif

#ifndef M_PI
/** An approximation of the constant pi. */
#define M_PI 3.141592654f
#endif

	/** A vector of four 32-bit floating-point components. */
	typedef
#ifdef VMATH_SSE_INTRINSICS
		__m128
#else
		struct {
			/** The components of the vector. */
			float v[4];
		}
#endif
	VECTOR;

	/**
	 * Stores a representation of the vector \a _A in the float array \a _V and returns \a _V.
	 * @def VectorGet(_V, _A)
	 * @param[out] _V A 4 elements long 16-byte aligned float array to store in.
	 * @param[in] _A The vector to be retrieved.
	 */
#ifdef VMATH_SSE_INTRINSICS
#define VectorGet(_V, _A) (_mm_store_ps(_V, _A), _V)
#else
#define VectorGet(_V, _A) (memcpy(_V, (_A).v, sizeof(float) * 4), _V)
#endif

	/**
	 * Returns a ::VECTOR, whoose components are solely \a v.
	 * @param v The value to use for the components.
	 */
	VMATH_INLINE VECTOR VectorReplicate(float v) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_set1_ps(v);
#else
		VECTOR result = { v, v, v, v };
		return result;
#endif
	}

/** A null/zero vector with the length 0. */
#ifdef VMATH_SSE_INTRINSICS
#define ZERO_VECTOR _mm_setzero_ps()
#else
#define ZERO_VECTOR VectorReplicate(0)
#endif

	/**
	 * Returns a ::VECTOR consisting of the components \a x, \a y, \a z and \a w.
	 * @param x The x component.
	 * @param y The y component.
	 * @param z The z component.
	 * @param w The w component.
	 * @return A vector of the specified components.
	 */
	VMATH_INLINE VECTOR VectorSet(float x, float y, float z, float w) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_set_ps(w, z, y, x);
#else
		VECTOR v = { x, y, z, w };
		return v;
#endif
	}

	/**
	 * Loads and returns a ::VECTOR from the float array \a v.
	 * @param v The float array to load.
	 */
	VMATH_INLINE VECTOR VectorLoad(float *v) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_load_ps(v);
#else
		VECTOR result = { v[0], v[1], v[2], v[3] };
		return result;
#endif
	}

	/** Adds the two vectors \a a and \a b (a + b).
	 * @param a The first vector to add.
	 * @param b The second vector to add.
	 * @return The sum of the two vectors.
	 */
	VMATH_INLINE VECTOR VectorAdd(VECTOR a, VECTOR b) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_add_ps(a, b);
#else
		VECTOR v = { a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2], a.v[3] + b.v[3] };
		return v;
#endif
	}

	/** Subtracts the vector \a b from \a a (a - b).
	 * @param a The vector to be subtracted.
	 * @param b The vector to subtract.
	 * @return The difference between the two vectors.
	 */
	VMATH_INLINE VECTOR VectorSubtract(VECTOR a, VECTOR b) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_sub_ps(a, b);
#else
		VECTOR v = { a.v[0] - b.v[0], a.v[1] - b.v[1], a.v[2] - b.v[2], a.v[3] - b.v[3] };
		return v;
#endif
	}

	/** Multiplies the two vectors \a a and \a b (a * b).
	 * @param a The first vector to multiply.
	 * @param b The second vector to multiply.
	 * @return The product of sum of the two vectors.
	 */
	VMATH_INLINE VECTOR VectorMultiply(VECTOR a, VECTOR b) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_mul_ps(a, b);
#else
		VECTOR v = { a.v[0] * b.v[0], a.v[1] * b.v[1], a.v[2] * b.v[2], a.v[3] * b.v[3] };
		return v;
#endif
	}

	/** Divides the vector \a a with \a b (a / b).
	 * @param a The dividend.
	 * @param b The divisor.
	 * @return The quotient of the two vectors.
	 */
	VMATH_INLINE VECTOR VectorDivide(VECTOR a, VECTOR b) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_div_ps(a, b);
#else
		VECTOR v = { a.v[0] / b.v[0], a.v[1] / b.v[1], a.v[2] / b.v[2], a.v[3] / b.v[3] };
		return v;
#endif
	}

	/** Computes the absolute value of each component of a vector.
	 * @param a The vector.
	 * @return A new vector with the absolute value of each component.
	 */
	VMATH_INLINE VECTOR VectorAbs(VECTOR a) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_andnot_ps(_mm_set1_ps(-0.0f), a);
#else
		VECTOR v = { fabsf(a.v[0]), fabsf(a.v[1]), fabsf(a.v[2]), fabsf(a.v[3]) };
		return v;
#endif
	}

	/**
	 * Returns the dot product, a.k.a. the scalar product, of the two vectors \a a and \a b.
	 */
	VMATH_INLINE float Vector3Dot(VECTOR a, VECTOR b) {
#if defined(VMATH_SSE4_1_INTRINSICS)
		return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x71));
#elif defined(VMATH_SSE_INTRINSICS)
		__m128 dot = _mm_mul_ps(a, b),
			   tmp = _mm_shuffle_ps(dot, dot, _MM_SHUFFLE(2, 1, 2, 1));
		dot = _mm_add_ss(dot, tmp);
		tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 1, 1, 1));
		return _mm_cvtss_f32(_mm_add_ss(dot, tmp));
#else
		return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2];
#endif
	}

	/**
	 * Returns the cross product, a.k.a. the vector product, of the two vectors \a a and \a b.
	 */
	VMATH_INLINE VECTOR VectorCross(VECTOR a, VECTOR b) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))), _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1))));
#else
		VECTOR v = { a.v[1] * b.v[2] - a.v[2] * b.v[1],
			a.v[2] * b.v[0] - a.v[0] * b.v[2],
			a.v[0] * b.v[1] - a.v[1] * b.v[0],
			0 };
		return v;
#endif
	}

	/**
	 * Returns the length or magnitude or norm of the vector \a a (||a||).
	 */
	VMATH_INLINE float Vector3Length(VECTOR a) {
#ifdef VMATH_SSE4_1_INTRINSICS
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(a, a, 0x71)));
#elif defined(VMATH_SSE_INTRINSICS)
		return sqrt(Vector3Dot(a, a));
#else
		return sqrt(a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2]);
#endif
	}

	/**
	 */
	VMATH_INLINE float Vector4Dot(VECTOR a, VECTOR b) {
#ifdef VMATH_SSE4_1_INTRINSICS
		return _mm_cvtss_f32(_mm_dp_ps(a, b, 0xFF));
#elif defined(VMATH_SSE3_INTRINSICS)
		__m128 tmp = _mm_mul_ps(a, b);
		tmp = _mm_hadd_ps(tmp, tmp);
		return _mm_cvtss_f32(_mm_hadd_ps(tmp, tmp));
#elif defined(VMATH_SSE_INTRINSICS)
		__m128 tmp1, tmp2;
		tmp1 = _mm_mul_ps(a, b);
		tmp2 = _mm_shuffle_ps(b, tmp1, _MM_SHUFFLE(1, 0, 0, 0));
		tmp2 = _mm_add_ps(tmp2, tmp1);
		tmp1 = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(0, 3, 0, 0));
		tmp1 = _mm_add_ps(tmp1, tmp2);
		return _mm_cvtss_f32(_mm_shuffle_ps(tmp1, tmp1, _MM_SHUFFLE(2, 2, 2, 2)));
#else
		return a.v[0] * b.v[0] + a.v[1] * b.v[1] + a.v[2] * b.v[2] + a.v[3] * b.v[3];
#endif
	}

	/**
	 * Returns the length or magnitude or norm of the vector \a a (||a||).
	 */
	VMATH_INLINE float Vector4Length(VECTOR a) {
#ifdef VMATH_SSE4_1_INTRINSICS
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(a, a, 0xFF)));
#elif defined(VMATH_SSE_INTRINSICS)
		return sqrt(Vector4Dot(a, a));
#else
		return sqrt(a.v[0] * a.v[0] + a.v[1] * a.v[1] + a.v[2] * a.v[2] + a.v[3] * a.v[3]);
#endif
	}

	/**
	 * Returns the normalized vector \a a.
	 * @param a The vector to normalize.
	 * @return The normalized vector \a a.
	 */
	VMATH_INLINE VECTOR Vector4Normalize(VECTOR a) {
#ifdef VMATH_SSE4_1_INTRINSICS
		return _mm_mul_ps(a, _mm_rsqrt_ps(_mm_dp_ps(a, a, 0xFF)));
#elif defined(VMATH_SSE3_INTRINSICS)
		__m128 tmp = _mm_mul_ps(a, b);
		tmp = _mm_hadd_ps(tmp, tmp);
		return _mm_cvtss_f32(_mm_hadd_ps(tmp, tmp));
#elif defined(VMATH_SSE_INTRINSICS)
		__m128 tmp = _mm_mul_ps(a, a);
		tmp = _mm_add_ps(tmp, _mm_shuffle_ps(tmp, tmp, 0x4E));
		return _mm_div_ps(a, _mm_sqrt_ps(_mm_add_ps(tmp, _mm_shuffle_ps(tmp, tmp, 0x11))));
#else
		float invLength = 1 / Vector4Length(a);
		VECTOR v = { a.v[0] * invLength, a.v[1] * invLength, a.v[2] * invLength, a.v[3] * invLength};
		return v;
#endif
	}

	/**
	 * Compare the elements in \a a and \a b and return a bit mask where \c 1 corresponds to equality.
	 * @param a The vector to compare.
	 * @param b The vector to compare.
	 * @return The result as a bit mask.
	 */
	VMATH_INLINE int VectorEqual(VECTOR a, VECTOR b) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_movemask_ps(_mm_cmpeq_ps(a, b));
#else
		return (a.v[0] == b.v[0] ? 1 : 0) | (a.v[1] == b.v[1] ? 1 << 1 : 0) | (a.v[2] == b.v[2] ? 1 << 2 : 0) | (a.v[3] == b.v[3] ? 1 << 3 : 0);
#endif
	}

	/**
	 * Constructs a new ::VECTOR from the Euler angles \a pitch, \a yaw and \a roll in radians.
	 * @param pitch The pitch in radians.
	 * @param yaw The yaw in radians.
	 * @param roll The roll in radians.
	 * @return A new ::VECTOR from the supplied angles.
	 */
	VMATH_INLINE VECTOR QuaternionRotationRollPitchYaw(float pitch, float yaw, float roll) {
		const float c1 = cos(pitch * 0.5),
			  s1 = sin(pitch * 0.5),
			  c2 = cos(yaw * 0.5),
			  s2 = sin(yaw * 0.5),
			  c3 = cos(roll * 0.5),
			  s3 = sin(roll * 0.5);
#ifdef VMATH_SSE_INTRINSICS
		return _mm_setr_ps(c2 * s1 * c3 + s2 * c1 * s3,
				s2 * c1 * c3 - c2 * s1 * s3,
				c2 * c1 * s3 - s2 * s1 * c3,
				c2 * c1 * c3 + s2 * s1 * s3);
#else
		VECTOR q = { c2 * s1 * c3 + s2 * c1 * s3,
			s2 * c1 * c3 - c2 * s1 * s3,
			c2 * c1 * s3 - s2 * s1 * c3,
			c2 * c1 * c3 + s2 * s1 * s3 };
		return q;
#endif
	}

	/** A 4x4 matrix. */
	typedef struct ALIGN(16) {
#ifdef VMATH_SSE_INTRINSICS
		__m128 r[4]; /**< The rows of the matrix. */
#else
		float m[16]; /**< The components of the matrix. */
#endif
	} MATRIX;

	/**
	 * Stores a representation of the matrix \a _A in the float array \a _V and returns \a _V.
	 * @def MatrixGet(_V, _A)
	 * @param[out] _V A 16 elements long 16-byte aligned float array to store in.
	 * @param[in] _A The matrix to be stored.
	 *
	 * The matrix is not passed as a pointer, contrary to the rest of the matrix functions.
	 */
#ifdef VMATH_SSE_INTRINSICS
#define MatrixGet(_V, _A) (_mm_store_ps((_V), (_A).r[0]), _mm_store_ps((_V) + 4, (_A).r[1]), _mm_store_ps((_V) + 8, (_A).r[2]), _mm_store_ps((_V) + 12, (_A).r[3]), (_V))
#else
#define MatrixGet(_V, _A) (memcpy((_V), (_A).m, sizeof(float) * 16), (_V))
#endif

	/**
	 * Creates a matrix from the specified components.
	 * @param m00 Element in column 0, row 0.
	 * @param m01 Element in column 0, row 1.
	 * @param m02 Element in column 0, row 2.
	 * @param m03 Element in column 0, row 3.
	 * @param m10 Element in column 1, row 0.
	 * @param m11 Element in column 1, row 1.
	 * @param m12 Element in column 1, row 2.
	 * @param m13 Element in column 1, row 3.
	 * @param m20 Element in column 2, row 0.
	 * @param m21 Element in column 2, row 1.
	 * @param m22 Element in column 2, row 2.
	 * @param m23 Element in column 2, row 3.
	 * @param m30 Element in column 3, row 0.
	 * @param m31 Element in column 3, row 1.
	 * @param m32 Element in column 3, row 2.
	 * @param m33 Element in column 3, row 3.
	 * @return The new matrix.
	 */
	VMATH_INLINE MATRIX MatrixSet(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33) {
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = { _mm_setr_ps(m00, m01, m02, m03), _mm_setr_ps(m10, m11, m12, m13), _mm_setr_ps(m20, m21, m22, m23), _mm_setr_ps(m30, m31, m32, m33) };
#else
		MATRIX m = { m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 };
#endif
		return m;
	}

	/**
	 * Loads and returns a ::MATRIX from the float array \a v.
	 * @param v The float array to load up.
	 * @return The matrix from the array.
	 */
	VMATH_INLINE MATRIX MatrixLoad(float *v) {
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = { _mm_load_ps(v), _mm_load_ps(v + 4), _mm_load_ps(v + 8), _mm_load_ps(v + 12) };
#else
		MATRIX m = { v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15] };
#endif
		return m;
	}

	/**
	 * Builds the identity matrix.
	 * @return The identity matrix.
	 */
	VMATH_INLINE MATRIX MatrixIdentity() {
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = { _mm_set_ps(0, 0, 0, 1), _mm_set_ps(0, 0, 1, 0), _mm_set_ps(0, 1, 0, 0), _mm_set_ps(1, 0, 0, 0) };
#else
		MATRIX m = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
#endif
		return m;
	}

	/**
	 * Computes the product of two matrices.
	 * @param a The first matrix to multiply.
	 * @param b The second matrix to multiply.
	 * @return The product of the two matrices.
	 * @warning The two matrices must be distinct, the result will be incorrect if \a a equals \a b.
	 */
	VMATH_INLINE MATRIX MatrixMultiply(MATRIX a, MATRIX b) {
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = {
			_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[0], b.r[0], 0x00), a.r[0]),
					_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[0], b.r[0], 0x55), a.r[1]),
						_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[0], b.r[0], 0xAA), a.r[2]),
							_mm_mul_ps(_mm_shuffle_ps(b.r[0], b.r[0], 0xFF), a.r[3])))),
			_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[1], b.r[1], 0x00), a.r[0]),
					_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[1], b.r[1], 0x55), a.r[1]),
						_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[1], b.r[1], 0xAA), a.r[2]),
							_mm_mul_ps(_mm_shuffle_ps(b.r[1], b.r[1], 0xFF), a.r[3])))),
			_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[2], b.r[2], 0x00), a.r[0]),
					_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[2], b.r[2], 0x55), a.r[1]),
						_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[2], b.r[2], 0xAA), a.r[2]),
							_mm_mul_ps(_mm_shuffle_ps(b.r[2], b.r[2], 0xFF), a.r[3])))),
			_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[3], b.r[3], 0x00), a.r[0]),
					_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[3], b.r[3], 0x55), a.r[1]),
						_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(b.r[3], b.r[3], 0xAA), a.r[2]),
							_mm_mul_ps(_mm_shuffle_ps(b.r[3], b.r[3], 0xFF), a.r[3]))))
		};
#else
		MATRIX m = {
			b.m[0] * a.m[0] + b.m[1] * a.m[4] + b.m[2] * a.m[8] + b.m[3] * a.m[12],
			b.m[0] * a.m[1] + b.m[1] * a.m[5] + b.m[2] * a.m[9] + b.m[3] * a.m[13],
			b.m[0] * a.m[2] + b.m[1] * a.m[6] + b.m[2] * a.m[10] + b.m[3] * a.m[14],
			b.m[0] * a.m[3] + b.m[1] * a.m[7] + b.m[2] * a.m[11] + b.m[3] * a.m[15],
			b.m[4] * a.m[0] + b.m[5] * a.m[4] + b.m[6] * a.m[8] + b.m[7] * a.m[12],
			b.m[4] * a.m[1] + b.m[5] * a.m[5] + b.m[6] * a.m[9] + b.m[7] * a.m[13],
			b.m[4] * a.m[2] + b.m[5] * a.m[6] + b.m[6] * a.m[10] + b.m[7] * a.m[14],
			b.m[4] * a.m[3] + b.m[5] * a.m[7] + b.m[6] * a.m[11] + b.m[7] * a.m[15],
			b.m[8] * a.m[0] + b.m[9] * a.m[4] + b.m[10] * a.m[8] + b.m[11] * a.m[12],
			b.m[8] * a.m[1] + b.m[9] * a.m[5] + b.m[10] * a.m[9] + b.m[11] * a.m[13],
			b.m[8] * a.m[2] + b.m[9] * a.m[6] + b.m[10] * a.m[10] + b.m[11] * a.m[14],
			b.m[8] * a.m[3] + b.m[9] * a.m[7] + b.m[10] * a.m[11] + b.m[11] * a.m[15],
			b.m[12] * a.m[0] + b.m[13] * a.m[4] + b.m[14] * a.m[8] + b.m[15] * a.m[12],
			b.m[12] * a.m[1] + b.m[13] * a.m[5] + b.m[14] * a.m[9] + b.m[15] * a.m[13],
			b.m[12] * a.m[2] + b.m[13] * a.m[6] + b.m[14] * a.m[10] + b.m[15] * a.m[14],
			b.m[12] * a.m[3] + b.m[13] * a.m[7] + b.m[14] * a.m[11] + b.m[15] * a.m[15]
		};
#endif
		return m;
	}

	/**
	 * Computes the transpose of a matrix.
	 * @param a The matrix to transpose.
	 * @return The transpose of \a a.
	 */
	VMATH_INLINE MATRIX MatrixTranspose(MATRIX a) {
#ifdef VMATH_SSE_INTRINSICS
		__m128 tmp0 = _mm_unpacklo_ps(a.r[0], a.r[1]),
			   tmp2 = _mm_unpacklo_ps(a.r[2], a.r[3]),
			   tmp1 = _mm_unpackhi_ps(a.r[0], a.r[1]),
			   tmp3 = _mm_unpackhi_ps(a.r[2], a.r[3]);
		MATRIX m = { _mm_movelh_ps(tmp0, tmp2), _mm_movehl_ps(tmp2, tmp0), _mm_movelh_ps(tmp1, tmp3), _mm_movehl_ps(tmp3, tmp1) };
#else
		MATRIX m = { a.m[0], a.m[4], a.m[8], a.m[12], a.m[1], a.m[5], a.m[9], a.m[13], a.m[2], a.m[6], a.m[10], a.m[14], a.m[3], a.m[7], a.m[11], a.m[15] };
#endif
		return m;
	}

	/**
	 * Inverses the matrix \a a using Cramer's rule.
	 * @param a The matrix to inverse.
	 * @return The inversed matrix \a a.
	 */
	VMATH_INLINE MATRIX MatrixInverse(MATRIX a) {
#ifdef VMATH_SSE_INTRINSICS
		__m128 minor0, minor1, minor2, minor3, row0, row1, row2, row3, det, tmp;
		a = MatrixTranspose(a);
		row0 = a.r[0];
		row1 = _mm_shuffle_ps(a.r[1], a.r[1], 0x4E);
		row2 = a.r[2];
		row3 = _mm_shuffle_ps(a.r[3], a.r[3], 0x4E);

		tmp = _mm_mul_ps(row2, row3);
		tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
		minor0 = _mm_mul_ps(row1, tmp);
		minor1 = _mm_mul_ps(row0, tmp);
		tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
		minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp), minor0);
		minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp), minor1);
		minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);

		tmp = _mm_mul_ps(row1, row2);
		tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
		minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp), minor0);
		minor3 = _mm_mul_ps(row0, tmp);
		tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
		minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp));
		minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp), minor3);
		minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);

		tmp = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
		tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
		row2 = _mm_shuffle_ps(row2, row2, 0x4E);
		minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp), minor0);
		minor2 = _mm_mul_ps(row0, tmp);
		tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
		minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp));
		minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp), minor2);
		minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);

		tmp = _mm_mul_ps(row0, row1);
		tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
		minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp), minor2);
		minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp), minor3);
		tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
		minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp), minor2);
		minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp));

		tmp = _mm_mul_ps(row0, row3);
		tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
		minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp));
		minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp), minor2);
		tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
		minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp), minor1);
		minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp));

		tmp = _mm_mul_ps(row0, row2);
		tmp = _mm_shuffle_ps(tmp, tmp, 0xB1);
		minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp), minor1);
		minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp));
		tmp = _mm_shuffle_ps(tmp, tmp, 0x4E);
		minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp));
		minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp), minor3);

		det = _mm_mul_ps(row0, minor0);
		det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
		det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
		tmp = _mm_rcp_ss(det);
		det = _mm_sub_ss(_mm_add_ss(tmp, tmp), _mm_mul_ss(det, _mm_mul_ss(tmp, tmp)));
		det = _mm_shuffle_ps(det, det, 0x00);
		MATRIX m = { _mm_mul_ps(det, minor0), _mm_mul_ps(det, minor1), _mm_mul_ps(det, minor2), _mm_mul_ps(det, minor3) };
#else
		float inv[] = { a.m[5] * a.m[10] * a.m[15] - a.m[5] * a.m[11] * a.m[14] - a.m[9] * a.m[6] * a.m[15] + a.m[9] * a.m[7] * a.m[14] + a.m[13] * a.m[6] * a.m[11] - a.m[13] * a.m[7] * a.m[10],
			-a.m[1] * a.m[10] * a.m[15] + a.m[1] * a.m[11] * a.m[14] + a.m[9] * a.m[2] * a.m[15] - a.m[9] * a.m[3] * a.m[14] - a.m[13] * a.m[2] * a.m[11] + a.m[13] * a.m[3] * a.m[10],
			a.m[1] * a.m[6] * a.m[15] - a.m[1] * a.m[7] * a.m[14] - a.m[5] * a.m[2] * a.m[15] + a.m[5] * a.m[3] * a.m[14] + a.m[13] * a.m[2] * a.m[7] - a.m[13] * a.m[3] * a.m[6],
			-a.m[1] * a.m[6] * a.m[11] + a.m[1] * a.m[7] * a.m[10] + a.m[5] * a.m[2] * a.m[11] - a.m[5] * a.m[3] * a.m[10] - a.m[9] * a.m[2] * a.m[7] + a.m[9] * a.m[3] * a.m[6],
			-a.m[4] * a.m[10] * a.m[15] + a.m[4] * a.m[11] * a.m[14] + a.m[8] * a.m[6] * a.m[15] - a.m[8] * a.m[7] * a.m[14] - a.m[12] * a.m[6] * a.m[11] + a.m[12] * a.m[7] * a.m[10],
			a.m[0] * a.m[10] * a.m[15] - a.m[0] * a.m[11] * a.m[14] - a.m[8] * a.m[2] * a.m[15] + a.m[8] * a.m[3] * a.m[14] + a.m[12] * a.m[2] * a.m[11] - a.m[12] * a.m[3] * a.m[10],
			-a.m[0] * a.m[6] * a.m[15] + a.m[0] * a.m[7] * a.m[14] + a.m[4] * a.m[2] * a.m[15] - a.m[4] * a.m[3] * a.m[14] - a.m[12] * a.m[2] * a.m[7] + a.m[12] * a.m[3] * a.m[6],
			a.m[0] * a.m[6] * a.m[11] - a.m[0] * a.m[7] * a.m[10] - a.m[4] * a.m[2] * a.m[11] + a.m[4] * a.m[3] * a.m[10] + a.m[8] * a.m[2] * a.m[7] - a.m[8] * a.m[3] * a.m[6],
			a.m[4] * a.m[9] * a.m[15] - a.m[4] * a.m[11] * a.m[13] - a.m[8] * a.m[5] * a.m[15] + a.m[8] * a.m[7] * a.m[13] + a.m[12] * a.m[5] * a.m[11] - a.m[12] * a.m[7] * a.m[9],
			-a.m[0] * a.m[9] * a.m[15] + a.m[0] * a.m[11] * a.m[13] + a.m[8] * a.m[1] * a.m[15] - a.m[8] * a.m[3] * a.m[13] - a.m[12] * a.m[1] * a.m[11] + a.m[12] * a.m[3] * a.m[9],
			a.m[0] * a.m[5] * a.m[15] - a.m[0] * a.m[7] * a.m[13] - a.m[4] * a.m[1] * a.m[15] + a.m[4] * a.m[3] * a.m[13] + a.m[12] * a.m[1] * a.m[7] - a.m[12] * a.m[3] * a.m[5],
			-a.m[0] * a.m[5] * a.m[11] + a.m[0] * a.m[7] * a.m[9] + a.m[4] * a.m[1] * a.m[11] - a.m[4] * a.m[3] * a.m[9] - a.m[8] * a.m[1] * a.m[7] + a.m[8] * a.m[3] * a.m[5],
			-a.m[4] * a.m[9] * a.m[14] + a.m[4] * a.m[10] * a.m[13] + a.m[8] * a.m[5] * a.m[14] - a.m[8] * a.m[6] * a.m[13] - a.m[12] * a.m[5] * a.m[10] + a.m[12] * a.m[6] * a.m[9],
			a.m[0] * a.m[9] * a.m[14] - a.m[0] * a.m[10] * a.m[13] - a.m[8] * a.m[1] * a.m[14] + a.m[8] * a.m[2] * a.m[13] + a.m[12] * a.m[1] * a.m[10] - a.m[12] * a.m[2] * a.m[9],
			-a.m[0] * a.m[5] * a.m[14] + a.m[0] * a.m[6] * a.m[13] + a.m[4] * a.m[1] * a.m[14] - a.m[4] * a.m[2] * a.m[13] - a.m[12] * a.m[1] * a.m[6] + a.m[12] * a.m[2] * a.m[5],
			a.m[0] * a.m[5] * a.m[10] - a.m[0] * a.m[6] * a.m[9] - a.m[4] * a.m[1] * a.m[10] + a.m[4] * a.m[2] * a.m[9] + a.m[8] * a.m[1] * a.m[6] - a.m[8] * a.m[2] * a.m[5] };

		float det = a.m[0] * inv[0] + a.m[1] * inv[4] + a.m[2] * inv[8] + a.m[3] * inv[12];
		if (det == 0) return a;
		det = 1.f / det;

		MATRIX m;
		for (int i = 0; i < 16; ++i) m.m[i] = inv[i] * det;
#endif
		return m;
	}

	/** Returns a perspective projection matrix.
	 * @param fov The field of vision in degrees.
	 * @param aspect The aspect ratio of the screen.
	 * @param zNear The near coordinate of the z-plane.
	 * @param zFar The far coordinate of the z-plane.
	 * @return The perspective matrix.
	 */
	VMATH_INLINE MATRIX MatrixPerspective(float fov, float aspect, float zNear, float zFar) {
		const float h = 1.0F / (float) tan(fov * M_PI / 360);
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = { _mm_setr_ps(h / aspect, 0, 0, 0), _mm_setr_ps(0, h, 0, 0), _mm_setr_ps(0, 0, (zNear + zFar) / (zNear - zFar), -1), _mm_setr_ps(0, 0, 2 * (zNear * zFar) / (zNear - zFar), 0) };
#else
		MATRIX m = { h / aspect, 0, 0, 0, 0, h, 0, 0, 0, 0, (zNear + zFar) / (zNear - zFar), -1, 0, 0, 2 * (zNear * zFar) / (zNear - zFar), 0 };
#endif
		return m;
	}

	/** Returns a orthographic projection matrix.
	 * @param left The coordinate for the left vertical clipping plane.
	 * @param right The coordinate for the right vertical clipping plane.
	 * @param bottom The coordinate for the bottom horizontal clipping plane.
	 * @param top The coordinate for the top horizontal clipping plane.
	 * @param nearVal The distance to the nearer depth clipping plane.
	 * @param farVal The distance to the farther depth clipping plane.
	 * @return The orthographic matrix.
	 */
	VMATH_INLINE MATRIX MatrixOrtho(float left, float right, float bottom, float top, float near, float far) {
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = { _mm_setr_ps(2 / (right - left), 0, 0, 0),
			_mm_setr_ps(0, 2 / (top - bottom), 0, 0),
			_mm_setr_ps(0, 0, -2 / (far - near), 0),
			_mm_setr_ps(-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1) };
#else
		MATRIX m = { 2 / (right - left), 0, 0, 0, 0, 2 / (top - bottom), 0, 0, 0, 0, -2 / (far - near), 0, -(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1 };
#endif
		return m;
	}

	/**
	 * Builds a translation matrix from the specified offsets.
	 * @param x The translation along the x-axis.
	 * @param y The translation along the y-axis.
	 * @param z The translation along the z-axis.
	 * @return The translation matrix.
	 */
	VMATH_INLINE MATRIX MatrixTranslation(float x, float y, float z) {
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = { _mm_set_ps(0, 0, 0, 1), _mm_set_ps(0, 0, 1, 0), _mm_set_ps(0, 1, 0, 0), _mm_set_ps(1, z, y, x) };
#else
		MATRIX m = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1 };
#endif
		return m;
	}

	/**
	 * Builds a translation matrix from a vector.
	 * @param v A vector describing the translations along the x-axis, y-axis, and z-axis.
	 * @return The translation matrix.
	 */
	VMATH_INLINE MATRIX MatrixTranslationFromVector(VECTOR v) {
#ifdef VMATH_SSE_INTRINSICS
		__m128 t = _mm_move_ss(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 1, 0, 3)), _mm_set1_ps(1));
		MATRIX m = { _mm_setr_ps(1, 0, 0, 0), _mm_setr_ps(0, 1, 0, 0), _mm_setr_ps(0, 0, 1, 0), _mm_shuffle_ps(t, t, _MM_SHUFFLE(0, 3, 2, 1)) };
#else
		MATRIX m = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, v.v[0], v.v[1], v.v[2], 1 };
#endif
		return m;
	}

	/**
	 * Builds a matrix that scales along the x-axis, y-axis, and z-axis.
	 * @param x Scaling factor along the x-axis.
	 * @param y Scaling factor along the x-axis.
	 * @param z Scaling factor along the x-axis.
	 * @return The scaling matrix.
	 */
	VMATH_INLINE MATRIX MatrixScaling(float x, float y, float z) {
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = { _mm_setr_ps(x, 0, 0, 0), _mm_setr_ps(0, y, 0, 0), _mm_setr_ps(0, 0, z, 0), _mm_setr_ps(0, 0, 0, 1) };
#else
		MATRIX m = { x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1 };
#endif
		return m;
	}

	/**
	 * Builds a rotation matrix from the quaternion \a a.
	 * @param a quaternion defining the rotation.
	 * @return The rotation matrix.
	 */
	VMATH_INLINE MATRIX MatrixRotationQuaternion(VECTOR a) {
		ALIGN(16) float q[4];
		VectorGet(q, a);
		const float xs = 2 * q[0], ys = 2 * q[1], zs = 2 * q[2],
		wx = q[3] * xs, wy = q[3] * ys, wz = q[3] * zs,
		xx = q[0] * xs, xy = q[0] * ys, xz = q[0] * zs,
		yy = q[1] * ys, yz = q[1] * zs, zz = q[2] * zs;
#ifdef VMATH_SSE_INTRINSICS
		MATRIX m = {
			_mm_setr_ps(1 - (yy + zz), xy + wz, xz - wy, 0),
			_mm_setr_ps(xy - wz, 1 - (xx + zz), yz + wx, 0),
			_mm_setr_ps(xz + wy, yz - wx, 1 - (xx + yy), 0),
			_mm_setr_ps(0, 0, 0, 1)
		};
#else
		MATRIX m = { 1 - (yy + zz), xy + wz, xz - wy, 0, xy - wz, 1 - (xx + zz), yz + wx, 0, xz + wy, yz - wx, 1 - (xx + yy), 0, 0, 0, 0, 1 };
#endif
		return m;
	}

	/**
	 * Compare the elements in \a a and \a b and return a bit mask where \c 1 corresponds to equality.
	 * @param a The first vector to compare.
	 * @param b The second vector to compare.
	 * @return The result as a bit mask.
	 */
	VMATH_INLINE int MatrixEqual(MATRIX *a, MATRIX *b) {
#ifdef VMATH_SSE_INTRINSICS
		return _mm_movemask_ps(_mm_cmpeq_ps(a->r[0], b->r[0])) | (_mm_movemask_ps(_mm_cmpeq_ps(a->r[1], b->r[1])) << 4) | (_mm_movemask_ps(_mm_cmpeq_ps(a->r[2], b->r[2])) << 8) | (_mm_movemask_ps(_mm_cmpeq_ps(a->r[3], b->r[3])) << 12);
#else
		return (a->m[0] == b->m[0] ? 1 : 0) | (a->m[1] == b->m[1] ? 1 << 1 : 0) | (a->m[2] == b->m[2] ? 1 << 2 : 0) | (a->m[3] == b->m[3] ? 1 << 3 : 0)
			| (a->m[4] == b->m[4] ? 1 << 4 : 0) | (a->m[5] == b->m[5] ? 1 << 5 : 0) | (a->m[6] == b->m[6] ? 1 << 6 : 0) | (a->m[7] == b->m[7] ? 1 << 7 : 0)
			| (a->m[8] == b->m[8] ? 1 << 8 : 0) | (a->m[9] == b->m[9] ? 1 << 9 : 0) | (a->m[10] == b->m[10] ? 1 << 10 : 0) | (a->m[11] == b->m[11] ? 1 << 11 : 0)
			| (a->m[12] == b->m[12] ? 1 << 12 : 0) | (a->m[13] == b->m[13] ? 1 << 13 : 0) | (a->m[14] == b->m[14] ? 1 << 14 : 0) | (a->m[15] == b->m[15] ? 1 << 15 : 0);
#endif
	}

#ifdef __cplusplus
}
#endif

#endif
