函数语法:

    SGEMM( TRANSA,  TRANSB,  M,  N,  K,  ALPHA,  A, LDA, B, LDB, BETA, C, LDC) 

功能：
matrix matrix multiply ( row major order)
```
| C C C C C |           | A A A |                      | C C C C C |                        
| C C C C C |           | A A A | |B B B B B|          | C C C C C |
| C C C C C | = alpha * | A A A | |B B B B B| + beta * | C C C C C |  
| C C C C C |           | A A A | |B B B B B|          | C C C C C |
| C C C C C |           | A A A |                      | C C C C C | 
```
int m = 5;
int n = 3;
int k = 5;
float aplha = 1.0;
float beta = 1.0;

sgemm('N', 'N', m, n, k, alpha, A, n, B, k, beta, C, k);

 
参数：

transa -- 数据类型: char；         功能：设定矩阵A是否转置, ‘N'或 'n' 为不转置, ‘T'或 't' 或'C'或 'c' 表示矩阵A需转置.
transb -- 数据类型: char；         功能：设定矩阵B是否转置, ‘N'或 'n' 为不转置, ‘T'或 't' 或'C'或 'c' 表示矩阵B需转置.
m        -- 数据类型: int；            功能：矩阵A和矩阵C的行数. 
n         -- 数据类型: int；            功能：矩阵B和矩阵C的列数. 
k         -- 数据类型: int；            功能：矩阵A和矩阵B的列数.
alpha  -- 数据类型: float；         功能：数乘系数.
a         -- 数据类型: float array；功能：保存矩阵A.
lda      -- 数据类型: int；            功能：矩阵A的递增步长.
b         -- 数据类型: float array；功能：保存矩阵B.
ldb      -- 数据类型: int；            功能：矩阵B的递增步长.
beta    -- 数据类型: float；         功能：数乘系数.
c          -- 数据类型: float array;  功能：保存矩阵C, 计算结果写入矩阵C.
ldc       -- 数据类型: int；           功能：矩阵C的递增步长

  
Fortran语言版sgemm
源代码：
```Fortran
      SUBROUTINE SGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
*     .. Scalar Arguments ..
      REAL ALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTER TRANSA,TRANSB
*     ..
*     .. Array Arguments ..
      REAL A(LDA,*),B(LDB,*),C(LDC,*)
*     ..
*
*  Purpose
*  =======
*
*  SGEMM  performs one of the matrix-matrix operations
*
*     C := alpha*op( A )*op( B ) + beta*C,
*
*  where  op( X ) is one of
*
*     op( X ) = X   or   op( X ) = X',
*
*  alpha and beta are scalars, and A, B and C are matrices, with op( A )
*  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
*
*  Arguments
*  ==========
*
*  TRANSA - CHARACTER*1.
*           On entry, TRANSA specifies the form of op( A ) to be used in
*           the matrix multiplication as follows:
*
*              TRANSA = 'N' or 'n',  op( A ) = A.
*
*              TRANSA = 'T' or 't',  op( A ) = A'.
*
*              TRANSA = 'C' or 'c',  op( A ) = A'.
*
*           Unchanged on exit.
*
*  TRANSB - CHARACTER*1.
*           On entry, TRANSB specifies the form of op( B ) to be used in
*           the matrix multiplication as follows:
*
*              TRANSB = 'N' or 'n',  op( B ) = B.
*
*              TRANSB = 'T' or 't',  op( B ) = B'.
*
*              TRANSB = 'C' or 'c',  op( B ) = B'.
*
*           Unchanged on exit.
*
*  M      - INTEGER.
*           On entry,  M  specifies  the number  of rows  of the  matrix
*           op( A )  and of the  matrix  C.  M  must  be at least  zero.
*           Unchanged on exit.
*
*  N      - INTEGER.
*           On entry,  N  specifies the number  of columns of the matrix
*           op( B ) and the number of columns of the matrix C. N must be
*           at least zero.
*           Unchanged on exit.
*
*  K      - INTEGER.
*           On entry,  K  specifies  the number of columns of the matrix
*           op( A ) and the number of rows of the matrix op( B ). K must
*           be at least  zero.
*           Unchanged on exit.
*
*  ALPHA  - REAL            .
*           On entry, ALPHA specifies the scalar alpha.
*           Unchanged on exit.
*
*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is
*           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
*           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
*           part of the array  A  must contain the matrix  A,  otherwise
*           the leading  k by m  part of the array  A  must contain  the
*           matrix A.
*           Unchanged on exit.
*
*  LDA    - INTEGER.
*           On entry, LDA specifies the first dimension of A as declared
*           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
*           LDA must be at least  max( 1, m ), otherwise  LDA must be at
*           least  max( 1, k ).
*           Unchanged on exit.
*
*  B      - REAL             array of DIMENSION ( LDB, kb ), where kb is
*           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
*           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
*           part of the array  B  must contain the matrix  B,  otherwise
*           the leading  n by k  part of the array  B  must contain  the
*           matrix B.
*           Unchanged on exit.
*
*  LDB    - INTEGER.
*           On entry, LDB specifies the first dimension of B as declared
*           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
*           LDB must be at least  max( 1, k ), otherwise  LDB must be at
*           least  max( 1, n ).
*           Unchanged on exit.
*
*  BETA   - REAL            .
*           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
*           supplied as zero then C need not be set on input.
*           Unchanged on exit.
*
*  C      - REAL             array of DIMENSION ( LDC, n ).
*           Before entry, the leading  m by n  part of the array  C must
*           contain the matrix  C,  except when  beta  is zero, in which
*           case C need not be set on entry.
*           On exit, the array  C  is overwritten by the  m by n  matrix
*           ( alpha*op( A )*op( B ) + beta*C ).
*
*  LDC    - INTEGER.
*           On entry, LDC specifies the first dimension of C as declared
*           in  the  calling  (sub)  program.   LDC  must  be  at  least
*           max( 1, m ).
*           Unchanged on exit.
*
*
*  Level 3 Blas routine.
*
*  -- Written on 8-February-1989.
*     Jack Dongarra, Argonne National Laboratory.
*     Iain Duff, AERE Harwell.
*     Jeremy Du Croz, Numerical Algorithms Group Ltd.
*     Sven Hammarling, Numerical Algorithms Group Ltd.
*
*
*     .. External Functions ..
      LOGICAL LSAME
      EXTERNAL LSAME
*     ..
*     .. External Subroutines ..
      EXTERNAL XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC MAX
*     ..
*     .. Local Scalars ..
      REAL TEMP
      INTEGER I,INFO,J,L,NCOLA,NROWA,NROWB
      LOGICAL NOTA,NOTB
*     ..
*     .. Parameters ..
      REAL ONE,ZERO
      PARAMETER (ONE=1.0E+0,ZERO=0.0E+0)
*     ..
*
*     Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
*     transposed and set  NROWA, NCOLA and  NROWB  as the number of rows
*     and  columns of  A  and the  number of  rows  of  B  respectively.
*
      NOTA = LSAME(TRANSA,'N')
      NOTB = LSAME(TRANSB,'N')
      IF (NOTA) THEN
          NROWA = M
          NCOLA = K
      ELSE
          NROWA = K
          NCOLA = M
      END IF
      IF (NOTB) THEN
          NROWB = K
      ELSE
          NROWB = N
      END IF
*
*     Test the input parameters.
*
      INFO = 0
      IF ((.NOT.NOTA) .AND. (.NOT.LSAME(TRANSA,'C')) .AND.
     +    (.NOT.LSAME(TRANSA,'T'))) THEN
          INFO = 1
      ELSE IF ((.NOT.NOTB) .AND. (.NOT.LSAME(TRANSB,'C')) .AND.
     +         (.NOT.LSAME(TRANSB,'T'))) THEN
          INFO = 2
      ELSE IF (M.LT.0) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (K.LT.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,NROWA)) THEN
          INFO = 8
      ELSE IF (LDB.LT.MAX(1,NROWB)) THEN
          INFO = 10
      ELSE IF (LDC.LT.MAX(1,M)) THEN
          INFO = 13
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('SGEMM ',INFO)
          RETURN
      END IF
*
*     Quick return if possible.
*
      IF ((M.EQ.0) .OR. (N.EQ.0) .OR.
     +    (((ALPHA.EQ.ZERO).OR. (K.EQ.0)).AND. (BETA.EQ.ONE))) RETURN
*
*     And if  alpha.eq.zero.
*
      IF (ALPHA.EQ.ZERO) THEN
          IF (BETA.EQ.ZERO) THEN
              DO 20 J = 1,N
                  DO 10 I = 1,M
                      C(I,J) = ZERO
   10             CONTINUE
   20         CONTINUE
          ELSE
              DO 40 J = 1,N
                  DO 30 I = 1,M
                      C(I,J) = BETA*C(I,J)
   30             CONTINUE
   40         CONTINUE
          END IF
          RETURN
      END IF
*
*     Start the operations.
*
      IF (NOTB) THEN
          IF (NOTA) THEN
*
*           Form  C := alpha*A*B + beta*C.
*
              DO 90 J = 1,N
                  IF (BETA.EQ.ZERO) THEN
                      DO 50 I = 1,M
                          C(I,J) = ZERO
   50                 CONTINUE
                  ELSE IF (BETA.NE.ONE) THEN
                      DO 60 I = 1,M
                          C(I,J) = BETA*C(I,J)
   60                 CONTINUE
                  END IF
                  DO 80 L = 1,K
                      IF (B(L,J).NE.ZERO) THEN
                          TEMP = ALPHA*B(L,J)
                          DO 70 I = 1,M
                              C(I,J) = C(I,J) + TEMP*A(I,L)
   70                     CONTINUE
                      END IF
   80             CONTINUE
   90         CONTINUE
          ELSE
*
*           Form  C := alpha*A'*B + beta*C
*
              DO 120 J = 1,N
                  DO 110 I = 1,M
                      TEMP = ZERO
                      DO 100 L = 1,K
                          TEMP = TEMP + A(L,I)*B(L,J)
  100                 CONTINUE
                      IF (BETA.EQ.ZERO) THEN
                          C(I,J) = ALPHA*TEMP
                      ELSE
                          C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                      END IF
  110             CONTINUE
  120         CONTINUE
          END IF
      ELSE
          IF (NOTA) THEN
*
*           Form  C := alpha*A*B' + beta*C
*
              DO 170 J = 1,N
                  IF (BETA.EQ.ZERO) THEN
                      DO 130 I = 1,M
                          C(I,J) = ZERO
  130                 CONTINUE
                  ELSE IF (BETA.NE.ONE) THEN
                      DO 140 I = 1,M
                          C(I,J) = BETA*C(I,J)
  140                 CONTINUE
                  END IF
                  DO 160 L = 1,K
                      IF (B(J,L).NE.ZERO) THEN
                          TEMP = ALPHA*B(J,L)
                          DO 150 I = 1,M
                              C(I,J) = C(I,J) + TEMP*A(I,L)
  150                     CONTINUE
                      END IF
  160             CONTINUE
  170         CONTINUE
          ELSE
*
*           Form  C := alpha*A'*B' + beta*C
*
              DO 200 J = 1,N
                  DO 190 I = 1,M
                      TEMP = ZERO
                      DO 180 L = 1,K
                          TEMP = TEMP + A(L,I)*B(J,L)
  180                 CONTINUE
                      IF (BETA.EQ.ZERO) THEN
                          C(I,J) = ALPHA*TEMP
                      ELSE
                          C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                      END IF
  190             CONTINUE
  200         CONTINUE
          END IF
      END IF
*
      RETURN
*
*     End of SGEMM .
*
      END
```



C语言版sgemm

源代码：
```C
    /******************************************************************************/
    /*
      Purpose:
        SGEMM computes C = alpha * A * B and related operations.
      Discussion:
        SGEMM performs one of the matrix-matrix operations
         C := alpha * op ( A ) * op ( B ) + beta * C,
        where op ( X ) is one of
          op ( X ) = X   or   op ( X ) = X',
        ALPHA and BETA are scalars, and A, B and C are matrices, with op ( A )
        an M by K matrix, op ( B ) a K by N matrix and C an N by N matrix.
      Licensing:
        This code is distributed under the GNU LGPL license. 
        
        The GNU Lesser General Public License is a Free Software license. Like any Free Software license, 
        it grants to you the four following freedoms:
            The freedom to run the program for any purpose.
            The freedom to study how the program works and adapt it to specific needs.
            The freedom to redistribute copies so you can help your neighbor.
            The freedom to improve the program and release your improvements to the public, 
                so that the whole community benefits.
        You may exercise the freedoms specified here provided that you comply with the express conditions of 
        this license. The principal conditions are:
            You must conspicuously and appropriately publish on each copy distributed an appropriate copyright 
              notice and disclaimer of warranty and keep intact all the notices that refer to this License and 
              to the absence of any warranty; and give any other recipients of the Program a copy of 
              the GNU Lesser General Public License along with the Program. Any translation of 
              the GNU Lesser General Public License must be accompanied by the GNU Lesser General Public License.
            If you modify your copy or copies of the library or any portion of it, you may distribute the 
              resulting library provided you do so under the GNU Lesser General Public License. However, 
              programs that link to the library may be licensed under terms of your choice, so long as 
              the library itself can be changed. Any translation of the GNU Lesser General Public License 
              must be accompanied by the GNU Lesser General Public License.
            If you copy or distribute the library, you must accompany it with the complete corresponding 
              machine-readable source code or with a written offer, valid for at least three years, to 
              furnish the complete corresponding machine-readable source code. You need not provide 
              source code to programs which link to the library.
            Any of these conditions can be waived if you get permission from the copyright holder.
        Your fair use and other rights are in no way affected by the above.
      Modified:
          March, 2017
      Author:
        Original FORTRAN77 version by Jack Dongarra.
        C version by Chunfeng Yang and John Burkardt.
      Parameters:
        Input, char TRANSA, specifies the form of op( A ) to be used in
        the matrix multiplication as follows:
        'N' or 'n', op ( A ) = A.
        'T' or 't', op ( A ) = A'.
        'C' or 'c', op ( A ) = A'.
        Input, char TRANSB, specifies the form of op ( B ) to be used in
        the matrix multiplication as follows:
        'N' or 'n', op ( B ) = B.
        'T' or 't', op ( B ) = B'.
        'C' or 'c', op ( B ) = B'.
        Input, int M, the number of rows of the  matrix op ( A ) and of the
        matrix C.  0 <= M.
        Input, int N, the number  of columns of the matrix op ( B ) and the
        number of columns of the matrix C.  0 <= N.
        Input, int K, the number of columns of the matrix op ( A ) and the
        number of rows of the matrix op ( B ).  0 <= K.
        Input, float ALPHA, the scalar multiplier
        for op ( A ) * op ( B ).
        Input, float A(LDA,KA), where:
        if TRANSA is 'N' or 'n', KA is equal to K, and the leading M by K
        part of the array contains A;
        if TRANSA is not 'N' or 'n', then KA is equal to M, and the leading
        K by M part of the array must contain the matrix A.
        Input, int LDA, the first dimension of A as declared in the calling
        routine.  When TRANSA = 'N' or 'n' then LDA must be at least max ( 1, M ),
        otherwise LDA must be at least max ( 1, K ).
        Input, float B(LDB,KB), where:
        if TRANSB is 'N' or 'n', kB is N, and the leading K by N
        part of the array contains B;
        if TRANSB is not 'N' or 'n', then KB is equal to K, and the leading
        N by K part of the array must contain the matrix B.
        Input, int LDB, the first dimension of B as declared in the calling
        routine.  When TRANSB = 'N' or 'n' then LDB must be at least max ( 1, K ),
        otherwise LDB must be at least max ( 1, N ).
        Input, float BETA, the scalar multiplier for C.
        Input, float C(LDC,N).
        Before entry, the leading M by N part of this array must contain the
        matrix C, except when BETA is 0.0, in which case C need not be set
        on entry.
        On exit, the array C is overwritten by the M by N matrix
          alpha * op ( A ) * op ( B ) + beta * C.
        Input, int LDC, the first dimension of C as declared in the calling
        routine.  max ( 1, M ) <= LDC.
    */
    void cblas_sgemm(char transa, char transb, int m, int n, int k, float alpha,
    		float a[], int lda, float b[], int ldb, float beta, float c[], int ldc)
    {
     
    	//
    	//
    #ifdef DEBUG
    	printf("cblas_sgemm\n");
    #endif
     
    	int i;
    	int info;
    	int j;
    	int l;
    	int ncola;
    	int nrowa;
    	int nrowb;
    	int ncolb;
    	int nota;
    	int notb;
    	float temp;
    	/*
    	 Set NOTA and NOTB as true if A and B respectively are not
    	 transposed and set NROWA, NCOLA and NROWB as the number of rows
    	 and columns of A and the number of rows of B respectively.
    	 */
    	nota = ((transa == 'N') || (transa == 'n'));
     
    	if (nota)
    	{
    		nrowa = m;
    		ncola = k;
    	}
    	else
    	{
    		nrowa = k;
    		ncola = m;
    	}
     
    	notb = ((transb == 'N') || (transb == 'n'));
     
    	if (notb)
    	{
    		nrowb = k;
    		ncolb = n;
    	}
    	else
    	{
    		nrowb = n;
    		ncolb = k;
    	}
     
    #ifdef DEBUG
    	printf("cblas_sgemm: m  %d \n", m);
    	printf("cblas_sgemm: n  %d \n", n);
    	printf("cblas_sgemm: k  %d \n", k);
     
    	printf("cblas_sgemm: nrowa  %d \n", nrowa );
    	printf("cblas_sgemm: ncola  %d \n", ncola );
     
    	printf("cblas_sgemm: nrowb  %d \n", nrowb);
    	printf("cblas_sgemm: lda  %d \n", lda);
    	printf("cblas_sgemm: ldb  %d \n", ldb);
    	printf("cblas_sgemm: ldc  %d \n", ldc);
    #endif
     
    	/*
    	 Test the input parameters.
    	 */
    	info = 0;
     
    	if (!(transa == 'N' || transa == 'n' || transa == 'C' || transa == 'c'
    			|| transa == 'T' || transa == 't'))
    	{
    		fprintf(stderr, "\n");
    		fprintf(stderr, "SGEMM - Fatal error!\n");
    		fprintf(stderr, "  Input TRANSA had illegal value.\n");
    		exit(1);
    	}
     
    	if (!(transb == 'N' || transb == 'n' || transb == 'C' || transb == 'c'
    			|| transb == 'T' || transb == 't'))
    	{
    		fprintf(stderr, "\n");
    		fprintf(stderr, "SGEMM - Fatal error!\n");
    		fprintf(stderr, "  Input TRANSB had illegal value.\n");
    		exit(1);
    	}
     
    	if (m < 0)
    	{
    		fprintf(stderr, "\n");
    		fprintf(stderr, "SGEMM - Fatal error!\n");
    		fprintf(stderr, "  Input M had illegal value.\n");
    		exit(1);
    	}
     
    	if (n < 0)
    	{
    		fprintf(stderr, "\n");
    		fprintf(stderr, "SGEMM - Fatal error!\n");
    		fprintf(stderr, "  Input N had illegal value.\n");
    		exit(1);
    	}
     
    	if (k < 0)
    	{
    		fprintf(stderr, "\n");
    		fprintf(stderr, "SGEMM - Fatal error!\n");
    		fprintf(stderr, "  Input K had illegal value.\n");
    		exit(1);
    	}
     
    	if (lda < i4_max(1, ncola))
    	{
    		fprintf(stderr, "\n");
    		fprintf(stderr, "SGEMM - Fatal error!\n");
    		fprintf(stderr, "  Input LDA had illegal value.\n");
    		exit(1);
    	}
     
    	if (ldb < i4_max(1, ncolb))
    	{
    		fprintf(stderr, "\n");
    		fprintf(stderr, "SGEMM - Fatal error!\n");
    		fprintf(stderr, "  Input LDB had illegal value.\n");
    		exit(1);
    	}
     
    	if (ldc < i4_max(1, n))
    	{
    		fprintf(stderr, "\n");
    		fprintf(stderr, "SGEMM - Fatal error!\n");
    		fprintf(stderr, "  Input LDC had illegal value.\n");
    		exit(1);
    	}
     
    #ifdef DEBUG
    	printf(" A:\n");
    	for (int i = 0; i < m; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			printf( "%8f\t", a[i * lda + j] );
    		}
    		printf("\n");
    	}
    #endif
     
    	/*
    	 Quick return if possible.
    	 */
    	if (m == 0)
    	{
    		return;
    	}
     
    	if (n == 0)
    	{
    		return;
    	}
     
    	if ((alpha == 0.0 || k == 0) && (beta == 1.0))
    	{
    		return;
    	}
    	/*
    	 And if alpha is 0.0.
    	 */
    	if (alpha == 0.0)
    	{
    		if (beta == 0.0)
    		{
    			for (j = 0; j < n; j++)
    			{
    				for (i = 0; i < m; i++)
    				{
    					c[i * ldc + j] = 0.0;
    				}
    			}
    		}
    		else
    		{
    			for (j = 0; j < n; j++)
    			{
    				for (i = 0; i < m; i++)
    				{
    					c[i * ldc + j] = beta * c[i * ldc + j];
    				}
    			}
    		}
    		return;
    	}
    	/*
    	 Start the operations.
    	 */
    	if (notb)
    	{
    		/*
    		 Form  C := alpha*A*B + beta*C.
    		 */
    		if (nota)
    		{
    #ifdef DEBUG
    			printf("cblas_sgemm: A-N  B-N \n");
    			printf("cblas_sgemm: C := alpha*A*B + beta*C \n", beta);
    			printf("cblas_sgemm: beta = %f \n", beta);
    #endif
     
    			for (int i = 0; i < m; i++)
    			{
    #ifdef DEBUG
    				printf("  \n");
    				printf("cblas_sgemm:  row %d \n", i);
    #endif
    				if (beta == 0.0)
    				{
    					for (int j = 0; j < n; j++)
    					{
    						c[i * ldc + j] = 0.0;
    					}
    				}
    				else if (beta != 1.0)
    				{
    					for (int j = 0; j < n; j++)
    					{
    						c[i * ldc + j] = beta * c[i * ldc + j];
    					}
    				}
     
    				for (int l = 0; l < k; l++)
    				{
    #ifdef DEBUG
    					printf("  \n");
    					printf(" A[%d][%d] = %f \n", i, l, a[i * lda + l]);
    					printf("  \n");
    #endif
    					if (a[i * lda + l] != 0.0)
    					{
    						float tmp = alpha * a[i * lda + l];
    						for (int j = 0; j < k; j++)
    						{
    							c[i * ldc + j] += tmp * b[l * ldb + j];
    						}
    					}
    				}
    			}
     
    		}
    		/*
    		 Form  C := alpha*A'*B + beta*C
    		 */
     
    		else
    		{
    #ifdef DEBUG
    			printf("cblas_sgemm: A-T  B-N \n");
    			printf("cblas_sgemm: C := alpha*A'*B + beta*C \n");
    #endif
     
    			//-------------------------------------
     
    			for (int i = 0; i < m; i++)
    			{
     
    				if (beta == 0.0)
    				{
    					for (int j = 0; j < n; j++)
    					{
    						c[i * ldc + j] = 0.0;
    					}
    				}
    				else if (beta != 1.0)
    				{
    					for (int j = 0; j < n; j++)
    					{
    						c[i * ldc + j] = beta * c[i * ldc + j];
    					}
    				}
    				for (int l = 0; l < k; l++)
    				{
    					// printf("A[%d][%d] = %f  \n", i, l, a[l * lda + i] );
    					for (int j = 0; j < n; j++)
    					{
    						float temp = a[l * lda + i] * b[l * ldb + j];
    						c[i * ldc + j] += alpha * temp;
    					}
    				}
    			}
     
    		}
    	}
    	/*
    	 Form  C := alpha*A*B' + beta*C
    	 */
    	else
    	{
    #ifdef DEBUG
    		printf("cblas_sgemm: A-N  B-T \n");
    		printf("cblas_sgemm: C := alpha*A*B' + beta*C \n");
    #endif
     
    		if (nota)
    		{
     
    			for (j = 0; j < n; j++)
    			{
    				if (beta == 0.0)
    				{
    					for (i = 0; i < m; i++)
    					{
    						c[i * ldc + j] = 0.0;
    					}
    				}
    				else if (beta != 1.0)
    				{
    					for (i = 0; i < m; i++)
    					{
    						c[i * ldc + j] = beta * c[i * ldc + j];
    					}
    				}
     
    				for (l = 0; l < k; l++)
    				{
    					if (b[l * ldb + j] != 0.0)
    					{
    						temp = alpha * b[l + j * ldb];
    						for (i = 0; i < m; i++)
    						{
    							c[i * ldc + j] = c[i * ldc + j]
    									+ temp * a[i * lda + l];
    						}
    					}
    				}
    			}
     
    		}
    		/*
    		 Form  C := alpha*A'*B' + beta*C
    		 */
    		else
    		{
    #ifdef DEBUG
    			printf("cblas_sgemm: A-T  B-T \n");
    #endif
     
    			for (int i = 0; i < m; i++)
    			{
    				if (beta == 0.0)
    				{
    					for (int j = 0; j < n; j++)
    					{
    						c[i * ldc + j] = 0.0;
    					}
    				}
    				else if (beta != 1.0)
    				{
    					for (int j = 0; j < n; j++)
    					{
    						c[i * ldc + j] = beta * c[i * ldc + j];
    					}
    				}
     
    				for (int l = 0; l < k; l++)
    				{
    #ifdef DEBUG
    					//printf("A[%d][%d] = %f  \n", i, l, a[l * lda + i] );
    #endif
    					for (int j = 0; j < n; j++)
    					{
    						float temp = a[l * lda + i] * b[j * ldb + l];
    						c[i * ldc + j] += alpha * temp;
    					}
    				}
    			}
    		}
    	}
    	return;
    }
```



Testing scenario 1

```     
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
     
     
    int i4_min ( int i1, int i2 )
    {
      int value;
     
     
      if ( i1 < i2 )
      {
        value = i1;
      }
      else
      {
        value = i2;
      }
      return value;
    }
     
     
    void show_matrix(float *A, int m, int n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                printf("%2.5f ", A[i * n + j]);
            printf("\n");
        }
    }
     
     
    void SETMATRIX( float* M, int i, int j, int n, float v)
    {
        if(!M)
        {
            return;
        }
        M[i*n+j] = v;
    }
     
     
    void sgemmTest()
    {
    	int row = 3;   // parameter m in sgemm method
    	int k = 3;
    	int n = 3;
     
    	float *G = (float*) malloc(row * n * sizeof(float));
    	float *H = (float*) malloc(n * k * sizeof(float));
    	float *C = (float*) malloc(row * k * sizeof(float));
     
    	for (int i = 0; i < row; i++)
    	{
    		for (int j = 0; j < n; j++)
    		{
    			SETMATRIX(G, i, j, n, 1);
    		}
    		SETMATRIX(H, i, i, k, i);
    	}
     
    	for (int i = 0; i < k; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			SETMATRIX(H, i, j, k, 1.0);
    		}
    	}
     
    	//
    	// A
    	SETMATRIX(G, 0, 1, k, 2);
    	SETMATRIX(G, 0, 2, k, 3);
     
    	SETMATRIX(G, 1, 2, k, 5);
     
    	SETMATRIX(G, 2, 0, k, 2);
    	SETMATRIX(G, 2, 1, k, 4);
     
    	SETMATRIX(G, 3, 0, k, 3);
    	SETMATRIX(G, 3, 2, k, 7);
     
    	SETMATRIX(G, 4, 0, k, 5);
    	SETMATRIX(G, 4, 1, k, 0);
     
    	//
    	// B
    	SETMATRIX(H, 0, 1, k, 2);
    	SETMATRIX(H, 0, 2, k, 5);
     
    	SETMATRIX(H, 1, 0, k, 7);
    	SETMATRIX(H, 1, 1, k, 2);
     
    	SETMATRIX(H, 2, 1, k, 3);
     
     
    	//
    	// C
    	printf("  \n");
     
    	for (int i = 0; i < row; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			SETMATRIX(  C, i, j, k, 0.0);
    		}
    	}
     
    	//
    	// Showing
    	printf("A:\n");
    	show_matrix( G, row, n );
     
    	printf("B:\n");
    	show_matrix( H,  n, k );
     
    	printf("C:\n");
    	show_matrix( C,  row, k ); 
     
     
    	printf("  \n");
    	printf("C := alpha*A*B + beta*C \n");
     
     
        float alpha = 1.0;
        float beta = 0.0;
     
     
        char transa = 'N';
        char transb = 'N';
     
    	printf("  \n");
     	printf("cblas_sgemm( %c, %c, %i, %i, %i, %5.2f, A,  %i, B, %i, %5.2f, C, %i )\n", transa, transb, row, k, n, alpha, 
         n,  k, beta, k );    
    	//
    	//
    	cblas_sgemm(transa, transb, row, k, n, alpha, G, n, H, k, beta, C,
    			k);
        //  
    	show_matrix( C,  row, n );
     
     
    	//
    	//
    	for (int i = 0; i < row; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			SETMATRIX(  C, i, j, k, 0.0);
    		}
    	}
        transa = 'T';
        transb = 'N';
         //
    	printf("  \n");
     	printf("cblas_sgemm( %c, %c, %i, %i, %i, %5.2f, A,  %i, B, %i, %5.2f, C, %i )\n", transa, transb, row, k, n, alpha, 
         n,  k, beta, k );    
    	//
    	//
    	cblas_sgemm(transa, transb, row, k, n, alpha, G, n, H, k, beta, C,
    			k);
        //  
    	show_matrix( C,  row, n );
     
     
    	//
    	//
    	for (int i = 0; i < row; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			SETMATRIX(  C, i, j, k, 0.0);
    		}
    	}
     
     
        transa = 'N';
        transb = 'T';
         //
    	printf("  \n");
     	printf("cblas_sgemm( %c, %c, %i, %i, %i, %5.2f, A,  %i, B, %i, %5.2f, C, %i )\n", transa, transb, row, k, n, alpha, 
         n,  k, beta, k );    
    	//
    	//
    	cblas_sgemm(transa, transb, row, k, n, alpha, G, n, H, k, beta, C,
    			k);
        //  
    	show_matrix( C,  row, n );
     
     
    	//
    	//
    	for (int i = 0; i < row; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			SETMATRIX(  C, i, j, k, 0.0);
    		}
    	}
        transa = 'T';
        transb = 'T';
         //
    	printf("  \n");
     	printf("cblas_sgemm( %c, %c, %i, %i, %i, %5.2f, A,  %i, B, %i, %5.2f, C, %i )\n", transa, transb, row, k, n, alpha, 
         n,  k, beta, k );    
    	//
    	//
    	cblas_sgemm(transa, transb, row, k, n, alpha, G, n, H, k, beta, C,
    			k);
        //  
    	show_matrix( C,  row, n );
     
    	return;
    }
     
     
    int main()
    {
        sgemmTest();
        return 1;
    }
```
Results of testing scenario 1

 
A:
1.00000 2.00000 3.00000 
1.00000 1.00000 5.00000 
2.00000 4.00000 1.00000 
B:
5.00000 2.00000 5.00000 
7.00000 2.00000 1.00000 
1.00000 3.00000 1.00000 
C:
0.00000 0.00000 0.00000 
0.00000 0.00000 0.00000 
0.00000 0.00000 0.00000 
  
C := alpha*A*B + beta*C 
  
cblas_sgemm( N, N, 3, 3, 3,  1.00, A,  3, B, 3,  0.00, C, 3 )
22.00000 15.00000 10.00000 
17.00000 19.00000 11.00000 
39.00000 15.00000 15.00000 
  
cblas_sgemm( T, N, 3, 3, 3,  1.00, A,  3, B, 3,  0.00, C, 3 )
14.00000 10.00000 8.00000 
21.00000 18.00000 15.00000 
51.00000 19.00000 21.00000 
  
cblas_sgemm( N, T, 3, 3, 3,  1.00, A,  3, B, 3,  0.00, C, 3 )
24.00000 14.00000 10.00000 
32.00000 14.00000 9.00000 
23.00000 23.00000 15.00000 
  
cblas_sgemm( T, T, 3, 3, 3,  1.00, A,  3, B, 3,  0.00, C, 3 )
17.00000 11.00000 6.00000 
32.00000 20.00000 9.00000 
30.00000 32.00000 19.00000 


Testing scenario 2 
```C
    void sgemmTest()
    {
    	int row = 4;   // parameter m in sgemm method
    	int k = 3;
    	int n = 3;
     
    	float *G = (float*) malloc(row * n * sizeof(float));
    	float *H = (float*) malloc(n * k * sizeof(float));
    	float *C = (float*) malloc(row * k * sizeof(float));
     
    	for (int i = 0; i < row; i++)
    	{
    		for (int j = 0; j < n; j++)
    		{
    			SETMATRIX(G, i, j, n, 1);
    		}
    		SETMATRIX(H, i, i, k, i);
    	}
     
    	for (int i = 0; i < k; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			SETMATRIX(H, i, j, k, 1.0);
    		}
    	}
     
    	//
    	// A
    	SETMATRIX(G, 0, 1, k, 2);
    	SETMATRIX(G, 0, 2, k, 3);
     
    	SETMATRIX(G, 1, 2, k, 5);
     
    	SETMATRIX(G, 2, 0, k, 2);
    	SETMATRIX(G, 2, 1, k, 4);
     
    	SETMATRIX(G, 3, 0, k, 3);
    	SETMATRIX(G, 3, 2, k, 7);
     
    	SETMATRIX(G, 4, 0, k, 5);
    	SETMATRIX(G, 4, 1, k, 0);
     
    	//
    	// B
    	SETMATRIX(H, 0, 1, k, 2);
    	SETMATRIX(H, 0, 2, k, 5);
     
    	SETMATRIX(H, 1, 0, k, 7);
    	SETMATRIX(H, 1, 1, k, 2);
     
    	SETMATRIX(H, 2, 1, k, 3);
     
    //	SETMATRIX(H, 3, 1, k, 2);
     
     
    	//
    	// C
    	printf("  \n");
     
    	for (int i = 0; i < row; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			SETMATRIX(  C, i, j, k, 0.0);
    		}
    	}
     
    	//
    	// Showing
    	printf("A:\n");
    	show_matrix( G, row, n );
     
    	printf("B:\n");
    	show_matrix( H,  n, k );
     
     
     
    	//
    	//
    	cblas_sgemm('N', 'N', row, k, n, 1.0, G, n, H, k, 0.0, C,
    			k);
     
    	printf("  \n");
    	printf("C := alpha*A*B + beta*C \n");
    	show_matrix( C,  row, n );
     
    	printf("  \n");
    	 printf("C := alpha*A'*B' + beta*C \n");
    	show_matrix( C,  row, n );
     
     
    	//
    	//
    	for (int i = 0; i < row; i++)
    	{
    		for (int j = 0; j < k; j++)
    		{
    			SETMATRIX(  C, i, j, k, 0.0);
    		}
    	}
    	cblas_sgemm('N', 'T', row, k, n, 1.0, G, n, H, k, 0.0, C,
    			k);
     
    	printf("  \n");
    	 printf("C := alpha*A *B' + beta*C \n");
    	show_matrix( C,  row, n );
     
    	return;
    }
```
Results of testing scenario 2 

BLAS testing

A:
1.000000 2.000000 3.000000
1.000000 1.000000 5.000000
2.000000 4.000000 1.000000
3.000000 1.000000 7.000000
B:
1.000000 2.000000 5.000000
7.000000 2.000000 1.000000
1.000000 3.000000 1.000000

C := alpha*A*B + beta*C
18.000000 15.000000 10.000000
13.000000 19.000000 11.000000
31.000000 15.000000 15.000000
17.000000 29.000000 23.000000

C := alpha*A *B' + beta*C
20.000000 14.000000 10.000000
28.000000 14.000000 9.000000
15.000000 23.000000 15.000000
40.000000 30.000000 13.000000


辅助函数
```C
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

int i4_min ( int i1, int i2 )
{
  int value;

  if ( i1 < i2 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}

void showVector(char* A, int n)
{
	for (int i = 0; i < n; i++)
	{
		printf("%2.5f ", A[i]);
	}
	printf("\n");
}

void showMatrix(float *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%2.5f ", A[i * n + j]);
        printf("\n");
    }
}

void showMatrix(float *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%2.5f ", A[i * n + j]);
        printf("\n");
    }
}
```