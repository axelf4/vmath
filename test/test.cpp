#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vmath.h>
#include <iostream>

namespace {

	TEST(Vector, VectorGet_simpleValues_ReturnsFloatArray) {
		ALIGN(16) float v[4];
		float *x = VectorGet(v, VectorSet(1, 2, 3, 4));
		float y[] = { 1, 2, 3, 4 };
		for (int i = 0; i < 4; ++i) {
			EXPECT_FLOAT_EQ(x[i], y[i]) << "Vectors x and y differs.";
		}
	}

	TEST(Vector, VectorEqual_EqualVectors_ReturnsEqual) {
		EXPECT_TRUE(VectorEqual(VectorSet(0, 0, 0, 0), VectorSet(0, 0, 0, 0)) == 0xF) << "Two equal vectors are reported as unequal.";
	}

	TEST(Vector, VectorEqual_UnequalVectors_ReturnsUnequal) {
		EXPECT_FALSE(VectorEqual(VectorSet(0, 0, 0, 0), VectorSet(0, 1, 1, 1)) == 0xF) << "Two unequal vectors are reported as equal.";
	}

	TEST(Vector, VectorReplicate_simpleValue_Calculated) {
		EXPECT_TRUE(VectorEqual(VectorReplicate(1), VectorSet(1, 1, 1, 1)) == 0xF);
	}

	TEST(Vector, ZERO_VECTOR_evaluation_Calculated) {
		EXPECT_TRUE(VectorEqual(ZERO_VECTOR, VectorReplicate(0)) == 0xF);
	}

	TEST(Vector, VectorSet_simpleValues_Calculated) {
		EXPECT_TRUE(VectorEqual(VectorSet(1, 2, 3, 4), VectorSet(1, 2, 3, 4)) == 0xF) << "The same vectors differ.";
	}

	TEST(Vector, VectorLoad_FloatArray_Calculated) {
		ALIGN(16) float v[] = { 1, 2, 3, 4 };
		EXPECT_TRUE(VectorEqual(VectorLoad(v), VectorSet(1, 2, 3, 4)) == 0xF);
	}

	TEST(Vector, VectorAdd_simpleValues_Calculated) {
		EXPECT_TRUE(VectorEqual(VectorAdd(VectorSet(1, 2, 3, 4), VectorSet(5, 6, 7, 8)), VectorSet(1 + 5, 2 + 6, 3 + 7, 4 + 8)) == 0xF) << "Result of vector addition differs.";
	}

	TEST(Vector, VectorSubtract_simpleValues_Calculated) {
		EXPECT_TRUE(VectorEqual(VectorSubtract(VectorSet(1, 2, 3, 4), VectorSet(5, 6, 7, 8)), VectorSet(1 - 5, 2 - 6, 3 - 7, 4 - 8)) == 0xF) << "Result of vector subtraction differs.";
	}

	TEST(Vector, VectorMultiply_simpleValues_Calculated) {
		EXPECT_TRUE(VectorEqual(VectorMultiply(VectorSet(1, 2, 3, 4), VectorSet(5, 6, 7, 8)), VectorSet(1 * 5, 2 * 6, 3 * 7, 4 * 8)) == 0xF) << "Result of vector multiplication differs.";
	}

	TEST(Vector, VectorDivide_simpleValues_Calculated) {
		EXPECT_TRUE(VectorEqual(VectorDivide(VectorSet(1, 2, 3, 4), VectorSet(5, 6, 7, 8)), VectorSet(1.f / 5, 2.f / 6, 3.f / 7, 4.f / 8)) == 0xF) << "Result of vector division differs.";
	}

	TEST(Vector, VectorAbs_simpleValues_Calculated) {
		EXPECT_TRUE(VectorEqual(VectorAbs(VectorSet(2, -1, -4, 0)), VectorSet(2, 1, 4, 0)) == 0xF);
	}

	TEST(Vector, Vector3Dot_simpleValues_Calculated) {
		EXPECT_FLOAT_EQ(Vector3Dot(VectorSet(1, 2, 3, 4), VectorSet(5, 6, 7, 8)), 1 * 5 + 2 * 6 + 3 * 7) << "Wrong dot product.";
	}

	TEST(Vector, Vector3Cross_simpleValues_Calculated) {
		EXPECT_TRUE(VectorEqual(Vector3Cross(VectorSet(1, 2, 3, 4), VectorSet(5, 6, 7, 8)), VectorSet(-4, 8, -4, 0)) == 0xF) << "The cross product of the vector differs.";
	}

	TEST(Vector, Vector3Length_simpleValues_Calculated) {
		float x = 1, y = 2, z = 3, w = 4;
		EXPECT_FLOAT_EQ(Vector3Length(VectorSet(1, 2, 3, 4)), sqrt(1 * 1 + 2 * 2 + 3 * 3)) << "The length of the vector is wrong.";
	}

	TEST(Vector, Vector4Dot_simpleValues_Calculated) {
		EXPECT_FLOAT_EQ(Vector4Dot(VectorSet(1, 2, 3, 4), VectorSet(5, 6, 7, 8)), 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8) << "Wrong dot product.";
	}

	TEST(Vector, Vector4Normalize_simpleValues_Calculated) {
		VECTOR v = VectorSet(1, 2, 3, 4);
		float length = Vector4Length(v);
		ALIGN(16) float a[4];
		VectorGet(a, Vector4Normalize(v));
		EXPECT_NEAR(a[0], 1.0f / length, 0.001) << "The normalized x component differs.";
		EXPECT_NEAR(a[1], 2.0f / length, 0.001) << "The normalized y component differs.";
		EXPECT_NEAR(a[2], 3.0f / length, 0.001) << "The normalized z component differs.";
		EXPECT_NEAR(a[3], 4.0f / length, 0.001) << "The normalized w component differs.";
	}

	bool matricesEqual(MATRIX *a, MATRIX *b) {
		ALIGN(16) float v0[16], v1[16];
		MatrixGet(v0, *a);
		MatrixGet(v1, *b);
		for (int i = 0; i < 16; ++i) {
			if (fabs(v0[i] - v1[i]) >= 0.00001)
				return false;
		}
		return true;
	}

	TEST(Matrix, Equal_SameComponents_True) {
		MATRIX m1 = MatrixSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), m2 = MatrixSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
		EXPECT_TRUE(MatrixEqual(&m1, &m2) == 0xFFFF);
	}

	TEST(Matrix, Equal_DifferentComponents_False) {
		MATRIX m1 = MatrixSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), m2 = MatrixSet(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
		EXPECT_FALSE(MatrixEqual(&m1, &m2) == 0xFFFF);
	}

	TEST(Matrix, MatrixLoad_simpleValues_Calculated) {
		ALIGN(16) float m[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
		MATRIX a = MatrixLoad(m), b = MatrixSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
		EXPECT_TRUE(MatrixEqual(&a, &b) == 0xFFFF);
	}

	TEST(Matrix, MatrixIdentity_simpleValues_ReturnsIdentity) {
		MATRIX a = MatrixIdentity(), b = MatrixSet(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		EXPECT_TRUE(MatrixEqual(&a, &b) == 0xFFFF);
	}

	TEST(Matrix, MatrixTranspose_simpleValues_Calculated) {
		MATRIX a = MatrixTranspose(MatrixSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)), b = MatrixSet(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16);
		EXPECT_TRUE(MatrixEqual(&a, &b) == 0xFFFF);
	}

	TEST(Matrix, MatrixMultiply_simpleValues_Calculated) {
		MATRIX m1 = MatrixSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), m2 = MatrixSet(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
		MATRIX a = MatrixMultiply(m1, m2), b = MatrixSet(80, 70, 60, 50, 240, 214, 188, 162, 400, 358, 316, 274, 560, 502, 444, 386);
		EXPECT_TRUE(MatrixEqual(&a, &b) == 0xFFFF);
	}

	TEST(Matrix, MatrixInverse_simpleValues_Calculated) {
		MATRIX a = MatrixSet(2, 3, 1, 5, 1, 0, 3, 1, 0, 2, -3, 2, 0, 2, 3, 1), b = MatrixSet(18, -35, -28, 1, 9, -18, -14, 1, -2, 4, 3, 0, -12, 24, 19, -1);
		MATRIX inverse = MatrixInverse(a);
		EXPECT_TRUE(matricesEqual(&inverse, &b));
	}
}
