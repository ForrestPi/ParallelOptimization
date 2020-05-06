## 函数语法: 

    XGEMV( TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY)

## 功能：

    matrix vector multiply

    BLAS level 2 function


C语言版(f2c)DGEMV
源代码：
```C
    /*  -- translated by f2c (version 19940927).
       You must link the resulting object file with the libraries:
    	-lf2c -lm   (in that order)
    */
     
    #include "f2c.h"
     
    /* Subroutine */ int dgemv_(char *trans, integer *m, integer *n, doublereal *
    	alpha, doublereal *a, integer *lda, doublereal *x, integer *incx, 
    	doublereal *beta, doublereal *y, integer *incy)
    {
     
     
        /* System generated locals */
        integer a_dim1, a_offset, i__1, i__2;
     
        /* Local variables */
        static integer info;
        static doublereal temp;
        static integer lenx, leny, i, j;
        extern logical lsame_(char *, char *);
        static integer ix, iy, jx, jy, kx, ky;
        extern /* Subroutine */ int xerbla_(char *, integer *);
     
     
    /*  Purpose   
        =======   
        DGEMV  performs one of the matrix-vector operations   
           y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   
        where alpha and beta are scalars, x and y are vectors and A is an   
        m by n matrix.   
        Parameters   
        ==========   
        TRANS  - CHARACTER*1.   
                 On entry, TRANS specifies the operation to be performed as   
                 follows:   
                    TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.   
                    TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.   
                    TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.   
                 Unchanged on exit.   
        M      - INTEGER.   
                 On entry, M specifies the number of rows of the matrix A.   
                 M must be at least zero.   
                 Unchanged on exit.   
        N      - INTEGER.   
                 On entry, N specifies the number of columns of the matrix A. 
      
                 N must be at least zero.   
                 Unchanged on exit.   
        ALPHA  - DOUBLE PRECISION.   
                 On entry, ALPHA specifies the scalar alpha.   
                 Unchanged on exit.   
        A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).   
                 Before entry, the leading m by n part of the array A must   
                 contain the matrix of coefficients.   
                 Unchanged on exit.   
        LDA    - INTEGER.   
                 On entry, LDA specifies the first dimension of A as declared 
      
                 in the calling (sub) program. LDA must be at least   
                 max( 1, m ).   
                 Unchanged on exit.   
        X      - DOUBLE PRECISION array of DIMENSION at least   
                 ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'   
                 and at least   
                 ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.   
                 Before entry, the incremented array X must contain the   
                 vector x.   
                 Unchanged on exit.   
        INCX   - INTEGER.   
                 On entry, INCX specifies the increment for the elements of   
                 X. INCX must not be zero.   
                 Unchanged on exit.   
        BETA   - DOUBLE PRECISION.   
                 On entry, BETA specifies the scalar beta. When BETA is   
                 supplied as zero then Y need not be set on input.   
                 Unchanged on exit.   
        Y      - DOUBLE PRECISION array of DIMENSION at least   
                 ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'   
                 and at least   
                 ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.   
                 Before entry with BETA non-zero, the incremented array Y   
                 must contain the vector y. On exit, Y is overwritten by the 
      
                 updated vector y.   
        INCY   - INTEGER.   
                 On entry, INCY specifies the increment for the elements of   
                 Y. INCY must not be zero.   
                 Unchanged on exit.   
        Level 2 Blas routine.   
        -- Written on 22-October-1986.   
           Jack Dongarra, Argonne National Lab.   
           Jeremy Du Croz, Nag Central Office.   
           Sven Hammarling, Nag Central Office.   
           Richard Hanson, Sandia National Labs.   
           Test the input parameters.   
        
       Parameter adjustments   
           Function Body */
    #define X(I) x[(I)-1]
    #define Y(I) y[(I)-1]
     
    #define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]
     
        info = 0;
        if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! 
    	    lsame_(trans, "C")) {
    	info = 1;
        } else if (*m < 0) {
    	info = 2;
        } else if (*n < 0) {
    	info = 3;
        } else if (*lda < max(1,*m)) {
    	info = 6;
        } else if (*incx == 0) {
    	info = 8;
        } else if (*incy == 0) {
    	info = 11;
        }
        if (info != 0) {
    	xerbla_("DGEMV ", &info);
    	return 0;
        }
     
    /*     Quick return if possible. */
     
        if (*m == 0 || *n == 0 || *alpha == 0. && *beta == 1.) {
    	return 0;
        }
     
    /*     Set  LENX  and  LENY, the lengths of the vectors x and y, and set 
      
           up the start points in  X  and  Y. */
     
        if (lsame_(trans, "N")) {
    	lenx = *n;
    	leny = *m;
        } else {
    	lenx = *m;
    	leny = *n;
        }
        if (*incx > 0) {
    	kx = 1;
        } else {
    	kx = 1 - (lenx - 1) * *incx;
        }
        if (*incy > 0) {
    	ky = 1;
        } else {
    	ky = 1 - (leny - 1) * *incy;
        }
     
    /*     Start the operations. In this version the elements of A are   
           accessed sequentially with one pass through A.   
           First form  y := beta*y. */
     
        if (*beta != 1.) {
    	if (*incy == 1) {
    	    if (*beta == 0.) {
    		i__1 = leny;
    		for (i = 1; i <= leny; ++i) {
    		    Y(i) = 0.;
    /* L10: */
    		}
    	    } else {
    		i__1 = leny;
    		for (i = 1; i <= leny; ++i) {
    		    Y(i) = *beta * Y(i);
    /* L20: */
    		}
    	    }
    	} else {
    	    iy = ky;
    	    if (*beta == 0.) {
    		i__1 = leny;
    		for (i = 1; i <= leny; ++i) {
    		    Y(iy) = 0.;
    		    iy += *incy;
    /* L30: */
    		}
    	    } else {
    		i__1 = leny;
    		for (i = 1; i <= leny; ++i) {
    		    Y(iy) = *beta * Y(iy);
    		    iy += *incy;
    /* L40: */
    		}
    	    }
    	}
        }
        if (*alpha == 0.) {
    	return 0;
        }
        if (lsame_(trans, "N")) {
     
    /*        Form  y := alpha*A*x + y. */
     
    	jx = kx;
    	if (*incy == 1) {
    	    i__1 = *n;
    	    for (j = 1; j <= *n; ++j) {
    		if (X(jx) != 0.) {
    		    temp = *alpha * X(jx);
    		    i__2 = *m;
    		    for (i = 1; i <= *m; ++i) {
    			Y(i) += temp * A(i,j);
    /* L50: */
    		    }
    		}
    		jx += *incx;
    /* L60: */
    	    }
    	} else {
    	    i__1 = *n;
    	    for (j = 1; j <= *n; ++j) {
    		if (X(jx) != 0.) {
    		    temp = *alpha * X(jx);
    		    iy = ky;
    		    i__2 = *m;
    		    for (i = 1; i <= *m; ++i) {
    			Y(iy) += temp * A(i,j);
    			iy += *incy;
    /* L70: */
    		    }
    		}
    		jx += *incx;
    /* L80: */
    	    }
    	}
        } else {
     
    /*        Form  y := alpha*A'*x + y. */
     
    	jy = ky;
    	if (*incx == 1) {
    	    i__1 = *n;
    	    for (j = 1; j <= *n; ++j) {
    		temp = 0.;
    		i__2 = *m;
    		for (i = 1; i <= *m; ++i) {
    		    temp += A(i,j) * X(i);
    /* L90: */
    		}
    		Y(jy) += *alpha * temp;
    		jy += *incy;
    /* L100: */
    	    }
    	} else {
    	    i__1 = *n;
    	    for (j = 1; j <= *n; ++j) {
    		temp = 0.;
    		ix = kx;
    		i__2 = *m;
    		for (i = 1; i <= *m; ++i) {
    		    temp += A(i,j) * X(ix);
    		    ix += *incx;
    /* L110: */
    		}
    		Y(jy) += *alpha * temp;
    		jy += *incy;
    /* L120: */
    	    }
    	}
        }
     
        return 0;
     
    /*     End of DGEMV . */
     
    } /* dgemv_ */
     



C语言版DGEMV

源代码：
```C
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
    
#include <windows.h>
#include <xmmintrin.h>
    
    
void ScaleValue1( float *pArray, DWORD dwCount, float fScale )
{
        DWORD dwGroupCount = dwCount / 4;
        __m128 e_Scale = _mm_set_ps1( fScale );
        for ( DWORD i = 0; i < dwGroupCount; i++ )
        {
                *(__m128*)( pArray + i * 4 ) = _mm_mul_ps( *(__m128*)( pArray + i * 4 ), e_Scale );
        }
}
    
void ScaleValue2( float *pArray, DWORD dwCount, float fScale )
{
        for ( DWORD i = 0; i < dwCount; i++ )
        {
                pArray[i] *= fScale;
        }
    
}
    
    
int i4_max ( int i1, int i2 )
{
    int value;
    
    if ( i2 < i1 )
    {
    value = i1;
    }
    else
    {
    value = i2;
    }
    return value;
}
    
    
int lsame ( char ca, char cb )
{
    if ( ca == cb )
    {
    return 1;
    }
    
    if ( 'A' <= ca && ca <= 'Z' )
    {
    if ( ca - 'A' == cb - 'a' )
    {
        return 1;
    }
    }
    else if ( 'a' <= ca && ca <= 'z' )
    {
    if ( ca - 'a' == cb - 'A' )
    {
        return 1;
    }
    }
    
    return 0;
}
    
    
    
void xerbla ( char *srname, int info )
{
    printf ( "\n" );
    printf ( "XERBLA - Fatal error!\n" );
    printf ( "  On entry to routine %s\n", srname );
    printf ( "  input parameter number %d had an illegal value.\n", info );
    return;  //exit ( 1 );
}
    
    
    
/******************************************************************************/
/*
    Purpose:
    SGEMV computes y := alpha * A * x + beta * y for general matrix A.
    Discussion:
    SGEMV performs one of the matrix-vector operations
        y := alpha*A *x + beta*y
    or
        y := alpha*A'*x + beta*y,
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.
    Licensing:
    This code is distributed under the GNU LGPL license.
    Modified:
    04 April 2014
    Author:
    C version by John Burkardt
    Parameters:
    Input, char TRANS, specifies the operation to be performed:
    'n' or 'N'   y := alpha*A *x + beta*y.
    't' or 'T'   y := alpha*A'*x + beta*y.
    'c' or 'C'   y := alpha*A'*x + beta*y.
    Input, int M, the number of rows of the matrix A.
    0 <= M.
    Input, int N, the number of columns of the matrix A.
    0 <= N.
    Input, float ALPHA, the scalar multiplier for A * x.
    Input, float A[LDA*N].  The M x N subarray contains
    the matrix A.
    Input, int LDA, the the first dimension of A as declared
    in the calling routine.  max ( 1, M ) <= LDA.
    Input, float X[*], an array containing the vector to be
    multiplied by the matrix A.
    If TRANS = 'N' or 'n', then X must contain N entries, stored in INCX
    increments in a space of at least ( 1 + ( N - 1 ) * abs ( INCX ) )
    locations.
    Otherwise, X must contain M entries, store in INCX increments
    in a space of at least ( 1 + ( M - 1 ) * abs ( INCX ) ) locations.
    Input, int INCX, the increment for the elements of
    X.  INCX must not be zero.
    Input, float BETA, the scalar multiplier for Y.
    Input/output, float Y[*], an array containing the vector to
    be scaled and incremented by A*X.
    If TRANS = 'N' or 'n', then Y must contain M entries, stored in INCY
    increments in a space of at least ( 1 + ( M - 1 ) * abs ( INCY ) )
    locations.
    Otherwise, Y must contain N entries, store in INCY increments
    in a space of at least ( 1 + ( N - 1 ) * abs ( INCY ) ) locations.
    Input, int INCY, the increment for the elements of
    Y.  INCY must not be zero.
*/
//
// row primary order version
void yblas_sgemv(char trans, int m, int n, float alpha, float a[], int lda,
        float x[], int incx, float beta, float y[], int incy)
{
    int i;
    int info;
    int ix;
    int iy;
    int j;
    int jx;
    int jy;
    int kx;
    int ky;
    int lenx;
    int leny;
    float temp;
    /*
        Test the input parameters.
        */
    info = 0;
    if (!lsame(trans, 'N') && !lsame(trans, 'T') && !lsame(trans, 'C'))
    {
        info = 1;
    }
    else if (m < 0)
    {
        info = 2;
    }
    else if (n < 0)
    {
        info = 3;
    }
    else if (lda < i4_max(1, m))
    {
        info = 6;
    }
    else if (incx == 0)
    {
        info = 8;
    }
    else if (incy == 0)
    {
        info = 11;
    }
    
    if (info != 0)
    {
        xerbla("SGEMV", info);
        return;
    }
    /*
        Quick return if possible.
        */
    if ((m == 0) || (n == 0) || ((alpha == 0.0) && (beta == 1.0)))
    {
        return;
    }
    /*
        Set LENX and LENY, the lengths of the vectors x and y, and set
        up the start points in X and Y.
        */
    if (lsame(trans, 'N'))
    {
        lenx = n;
        leny = m;
    }
    else
    {
        lenx = m;
        leny = n;
    }
    
    if (0 < incx)
    {
        kx = 0;
    }
    else
    {
        kx = 0 - (lenx - 1) * incx;
    }
    
    if (0 < incy)
    {
        ky = 0;
    }
    else
    {
        ky = 0 - (leny - 1) * incy;
    }
    
    /*
        Start the operations. In this version the elements of A are
        accessed sequentially with one pass through A.
        First form  y := beta*y.
        */
    if (beta != 1.0)
    {
        if (incy == 1)
        {
            if (beta == 0.0)
            {
                for (i = 0; i < leny; i++)
                {
                    y[i] = 0.0;
                }
            }
            else
            {
                for (i = 0; i < leny; i++)
                {
                    y[i] = beta * y[i];
                }
            }
        }
        else
        {
            iy = ky;
            if (beta == 0.0)
            {
                for (i = 0; i < leny; i++)
                {
                    y[iy] = 0.0;
                    iy = iy + incy;
                }
            }
            else
            {
                for (i = 0; i < leny; i++)
                {
                    y[iy] = beta * y[iy];
                    iy = iy + incy;
                }
            }
        }
    }
    
    if (alpha == 0.0)
    {
        return;
    }
    /*
        Form y := alpha*A*x + y.
        */
    if (lsame(trans, 'N'))
    {
        jx = kx;
        iy = ky;
        if (incy == 1)
        {
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    if (x[j] != 0.0)
                    {
                        y[i] += alpha * x[j] * a[i * lda + j];
                    }
                }
            }
        }
        else
        {
            for (i = 0; i < m; i++)
            {
                jx = kx;
                for (j = 0; j < n; j++)
                {
                    if (x[jx] != 0.0)
                    {
                        temp = alpha * x[jx];
                        y[iy] = y[iy] + temp * a[i * lda + j];
                    }
                    jx = jx + incx;
                }
                iy = iy + incy;
            }
        }
    }
    /*
        Form y := alpha*A'*x + y.
        */
    else
    {
        jy = ky;
        jx = kx;
        if (incx == 1)
        {
            for (j = 0; j < n; j++)
            {
                if (x[jx] != 0.0)
                {
                    temp = alpha * x[jx];
                    for (i = 0; i < m; i++)
                    {
                        y[i] = y[i] + temp * a[i + j * lda];
                    }
                }
                jx = jx + incx;
            }
        }
        else
        {
            for (i = 0; i < m; i++)
            {
                if (x[jx] != 0.0)
                {
                    temp = alpha * x[jx];
                    iy = ky;
                    for (j = 0; j < n; j++)
                    {
                        y[iy] = y[iy] + temp * a[i * lda + j];
                        iy = iy + incy;
                    }
                }
                jx = jx + incx;
            }
        }
    }
    return;
}
    
    
    
    
    
//
// column primary order version
void sgemv ( char trans, int m, int n, float alpha, float a[], int lda,
    float x[], int incx, float beta, float y[], int incy )
{
    int i;
    int info;
    int ix;
    int iy;
    int j;
    int jx;
    int jy;
    int kx;
    int ky;
    int lenx;
    int leny;
    float temp;
    /*
        Test the input parameters.
        */
    info = 0;
    if (!lsame(trans, 'N') && !lsame(trans, 'T') && !lsame(trans, 'C'))
    {
        info = 1;
    }
    else if (m < 0)
    {
        info = 2;
    }
    else if (n < 0)
    {
        info = 3;
    }
    else if (lda < i4_max(1, m))
    {
        info = 6;
    }
    else if (incx == 0)
    {
        info = 8;
    }
    else if (incy == 0)
    {
        info = 11;
    }
    
    if (info != 0)
    {
        xerbla("SGEMV", info);
        return;
    }
    /*
        Quick return if possible.
        */
    if ((m == 0) || (n == 0) || ((alpha == 0.0) && (beta == 1.0)))
    {
        return;
    }
    /*
        Set LENX and LENY, the lengths of the vectors x and y, and set
        up the start points in X and Y.
        */
    if (lsame(trans, 'N'))
    {
        lenx = n;
        leny = m;
    }
    else
    {
        lenx = m;
        leny = n;
    }
    
    if (0 < incx)
    {
        kx = 0;
    }
    else
    {
        kx = 0 - (lenx - 1) * incx;
    }
    
    if (0 < incy)
    {
        ky = 0;
    }
    else
    {
        ky = 0 - (leny - 1) * incy;
    }
    /*
        Start the operations. In this version the elements of A are
        accessed sequentially with one pass through A.
        First form  y := beta*y.
        */
    if (beta != 1.0)
    {
        if (incy == 1)
        {
            if (beta == 0.0)
            {
                for (i = 0; i < leny; i++)
                {
                    y[i] = 0.0;
                }
            }
            else
            {
                for (i = 0; i < leny; i++)
                {
                    y[i] = beta * y[i];
                }
            }
        }
        else
        {
            iy = ky;
            if (beta == 0.0)
            {
                for (i = 0; i < leny; i++)
                {
                    y[iy] = 0.0;
                    iy = iy + incy;
                }
            }
            else
            {
                for (i = 0; i < leny; i++)
                {
                    y[iy] = beta * y[iy];
                    iy = iy + incy;
                }
            }
        }
    }
    
    if (alpha == 0.0)
    {
        return;
    }
    /*
        Form y := alpha*A*x + y.
        */
    if (lsame(trans, 'N'))
    {
        jx = kx;
        if (incy == 1)
        {
            for (j = 0; j < n; j++)
            {
                if (x[jx] != 0.0)
                {
                    temp = alpha * x[jx];
                    for (i = 0; i < m; i++)
                    {
                        y[i] = y[i] + temp * a[i + j * lda];
                    }
                }
                jx = jx + incx;
            }
        }
        else
        {
            for (j = 0; j < n; j++)
            {
                if (x[jx] != 0.0)
                {
                    temp = alpha * x[jx];
                    iy = ky;
                    for (i = 0; i < m; i++)
                    {
                        y[iy] = y[iy] + temp * a[i + j * lda];
                        iy = iy + incy;
                    }
                }
                jx = jx + incx;
            }
        }
    }
    /*
        Form y := alpha*A'*x + y.
        */
    else
    {
        jy = ky;
        if (incx == 1)
        {
            for (j = 0; j < n; j++)
            {
                temp = 0.0;
                for (i = 0; i < m; i++)
                {
                    temp = temp + a[i + j * lda] * x[i];
                }
                y[jy] = y[jy] + alpha * temp;
                jy = jy + incy;
            }
        }
        else
        {
            for (j = 0; j < n; j++)
            {
                temp = 0.0;
                ix = kx;
                for (i = 0; i < m; i++)
                {
                    temp = temp + a[i + j * lda] * x[ix];
                    ix = ix + incx;
                }
                y[jy] = y[jy] + alpha * temp;
                jy = jy + incy;
            }
        }
    }
    
    return;
}
    
    
void sgemv_test()
{
    unsigned int start,end;
    double cost;
    
    
    int n = 20;
    
    float *x = (float*) calloc(n, sizeof(float));
    float *b = (float*) calloc(n, sizeof(float));
    float *A = (float*) calloc(n * n, sizeof(float));
    float *p = (float*) calloc(n, sizeof(float));
    float *r = (float*) calloc(n, sizeof(float));
    
    A[0] = 4;
    A[1] = 1;
    A[2] = 1;
    A[3] = 3;
    
    b[0] = 0;
    b[1] = 0;
    
    x[0] = 2;
    x[1] = 1;
    
    LARGE_INTEGER m_nFreq;
    LARGE_INTEGER m_nBeginTime;
    LARGE_INTEGER nEndTime;
    QueryPerformanceFrequency(&m_nFreq);
    QueryPerformanceCounter(&m_nBeginTime);
    
    
    for( int i = 0; i < 1000000; i++ )
        sgemv('T', n, n, 1.0, A, n, x, 1, -1.0, b, 1);
    
    QueryPerformanceCounter(&nEndTime);
    LONGLONG t = (nEndTime.QuadPart-m_nBeginTime.QuadPart)*1000/m_nFreq.QuadPart;
    
    printf("time consumption = %I64d \n", (nEndTime.QuadPart-m_nBeginTime.QuadPart)*1000/m_nFreq.QuadPart );
    
    
    free(x);
    free(b);
    free(A);
    free(p);
    free(r);
    
}
    
int main()
{
    for( int i = 0; i < 5; i++ )
        sgemv_test();
    
    return 0;
}
```

## 测试

The purpose of this testing is comparing computational efficiency of two different BLAS implementations:
C language implementation and OpenBLAS implementation, 

OpenBLAS is an optimized BLAS library. Before the testing, I have thought that the OpenBLAS implementation should have better performance. However, the results violented my expectation. 
 
 


测试环境:

CPU:  Intel Xeon E5-2667 
RAM:  64G
Windows 7 64Bit Operation System
VS2015 x64 Native Tools Command Prompt

编译参数：

cl -o test.exe sgemv.cc


运行结果：

2阶矩阵运算

time consumption = 60
time consumption = 39
time consumption = 39
time consumption = 39
time consumption = 39


20阶矩阵运算

time consumption = 1193
time consumption = 1142
time consumption = 1142
time consumption = 1142
time consumption = 1142



使用编译优化项参数：

cl -o test.exe sgemv.cc /FA /O2

2阶矩阵运算

time consumption = 27
time consumption = 21
time consumption = 21
time consumption = 21
time consumption = 21

20阶矩阵运算

time consumption = 261
time consumption = 239
time consumption = 229
time consumption = 229
time consumption = 229


30阶矩阵运算

time consumption = 586
time consumption = 571
time consumption = 513
time consumption = 513
time consumption = 513


100阶矩阵运算

time consumption = 6889
time consumption = 6742
time consumption = 6840
time consumption = 6757
time consumption = 6748



OpenBLAS version sgemv function testing scenario
```c
    #include <stdio.h>
    #include <malloc.h>
    #include <stdlib.h>
     
    #include <windows.h>
    #include <xmmintrin.h>
    #include "cblas.h"
     
    void sgemv_test() {
        int n = 20;
     
        float *x = (float*) calloc(n, sizeof(float));
        float *b = (float*) calloc(n, sizeof(float));
        float *A = (float*) calloc(n * n, sizeof(float));
        float *p = (float*) calloc(n, sizeof(float));
        float *r = (float*) calloc(n, sizeof(float));
     
        A[0] = 4;
        A[1] = 1;
        A[2] = 1;
        A[3] = 3;
     
        b[0] = 0;
        b[1] = 0;
     
        x[0] = 2;
        x[1] = 1;
     
     
        LARGE_INTEGER m_nFreq;
        LARGE_INTEGER m_nBeginTime;
        LARGE_INTEGER nEndTime;
        QueryPerformanceFrequency(&m_nFreq); // 获取时钟周期
        QueryPerformanceCounter(&m_nBeginTime); // 获取时钟计数
     
    	for( int i = 0; i < 1000000; i++ )
    		cblas_sgemv(CblasColMajor, CblasNoTrans, n, n, 1.0, A, n, x, 1, -1.0, b, 1);
     
     
        QueryPerformanceCounter(&nEndTime);
     
        printf("time consumption = %lld \n", (nEndTime.QuadPart-m_nBeginTime.QuadPart)*1000/m_nFreq.QuadPart );
     
        free(x);
        free(b);
        free(A);
        free(p);
        free(r);
     
    }
     
     
     
    int main()
    {
    	for( int i = 0; i < 5; i++ )
    	sgemv_test();
     
    	return 0;
    }
```

### 测试环境:

CPU:  Intel Xeon E5-2667 
RAM:  64G
Windows 7 64Bit Operation System
VS2015 x63 Native Tools Command Prompt


编译参数：

   /w /EHsc /D_CRT_SECURE_NO_DEPRECATE


OpenBLAS库：

Binary packages for Windows x86/x86_64 [1]


运行结果：

2阶矩阵运算

time consumption = 58
time consumption = 35
time consumption = 35
time consumption = 35
time consumption = 35


 20阶矩阵运算

time consumption = 108
time consumption = 86
time consumption = 86
time consumption = 86
time consumption = 86


30阶矩阵运算

time consumption = 181
time consumption = 161
time consumption = 161
time consumption = 161
time consumption = 161

100阶矩阵运算 

time consumption = 79538
time consumption = 79645
time consumption = 77239
time consumption = 77965
time consumption = 80059



测试结果分析
本文仅进行了小阶数矩阵运算测试，测试结果显示：

对于2阶矩阵
使用OpenBLAS库运行测试算例所用时间为35微秒，

使用C语言版sgemv函数，运行测试算例所用时间为39微秒。

使用编译优化项后，运行测试算例所用时间为21微秒。


对于20阶矩阵
使用OpenBLAS库运行测试算例所用时间为86微秒，
使用C语言版sgemv函数，运行测试算例所用时间为1142微秒，

使用编译优化项后，运行测试算例所用时间为229微秒。


对于30阶矩阵
使用OpenBLAS库运行测试算例所用时间为161微秒，
C语言版sgemv函数，使用编译优化项后，运行测试算例所用时间为513微秒。

对于100阶矩阵
使用OpenBLAS库运行测试算例所用时间约为78000微秒。
C语言版sgemv函数，使用编译优化项后，运行测试算例所用时间约为6750微秒。



小维度矩阵运算中使用OpenBLAS的提升计算效率的效果不大。

随着矩阵阶数的提高，OpenBLAS中优化效果明显，但是矩阵阶数在100阶附近时，OpenBLAS库版计算时间急剧增长，计算效率呈断崖式下跌。


Reference 
[1]  http://sourceforge.net/projects/openblas/files
