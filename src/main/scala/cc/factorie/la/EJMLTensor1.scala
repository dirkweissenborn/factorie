package cc.factorie.la

import cc.factorie.util.{DenseDoubleSeq, DoubleSeq, IntSeq, RangeIntSeq}
import org.ejml.alg.dense.mult.VectorVectorMult
import org.ejml.ops.{NormOps, CommonOps}
import org.ejml.simple.SimpleMatrix

/**
 * Created by diwe01 on 01.08.14.
 */
trait EJMLTensor extends Tensor with DenseDoubleSeq {
  val matrix:SimpleMatrix
  override def update(i: Int, v: Double): Unit = matrix.set(i,v)
  override def isDense: Boolean = true
  override def zero(): Unit = matrix.zero()
  override def apply(i: Int): Double = matrix.get(i)
  override def *=(d: Double): Unit = CommonOps.scale(d,matrix.getMatrix)
  override def oneNorm: Double = NormOps.fastNormP(matrix.getMatrix,1.0)
  override def twoNormSquared: Double = { val f = matrix.normF(); f*f }
  override def +=(i: Int, incr: Double): Unit = matrix.set(i, apply(i) + incr)
  override def *=(i: Int, incr: Double): Unit = matrix.set(i, apply(i) * incr)
  override def +=(ds: DoubleSeq, factor: Double): Unit = ds match {
    case ejml2:EJMLTensor =>
      if(factor == 1.0) CommonOps.addEquals(matrix.getMatrix, ejml2.matrix.getMatrix)
      else CommonOps.addEquals(matrix.getMatrix,factor,ejml2.matrix.getMatrix)
    case _ => ds.foreachActiveElement((i,v) => +=(i,v))
  }
  override def /=(ds: DoubleSeq): Unit = ds match {
    case ejml2:EJMLTensor =>
      CommonOps.elementDiv(matrix.getMatrix, ejml2.matrix.getMatrix)
    case _ => super./=(ds)
  }
  override def *=(ds: DoubleSeq): Unit = ds match {
    case ejml2:EJMLTensor => CommonOps.elementMult(matrix.getMatrix, ejml2.matrix.getMatrix)
    case _ => ds.foreachActiveElement((i,v) => *=(i,v))
  }
  override def +=(d: Double): Unit = CommonOps.add(matrix.getMatrix,d)
  override def forallActiveElements(f: (Int, Double) => Boolean): Boolean = {
    val a = matrix.getMatrix.iterator(true, 0, 0, matrix.numRows() - 1, matrix.numCols() - 1)
    var res = true
    while (res && a.hasNext) {
      val n = a.next()
      res &&= f(a.getIndex, n)
    }
    res
  }
}

class EJMLTensor1(override val matrix:SimpleMatrix) extends EJMLTensor with Tensor1 {
  assert(matrix.isVector)
  def this(dim1:Int) = this(new SimpleMatrix(dim1,1))
  override def dim1: Int = matrix.numRows()
  override def activeDomain: IntSeq = new RangeIntSeq(0,dim1)
  override def activeDomainSize: Int = dim1

  override def *(f: Double): Tensor1 = new EJMLTensor1(matrix.scale(f))
  override def +(t: Tensor1): Tensor1 = t match {
    case ejml2:EJMLTensor1 => new EJMLTensor1(matrix.plus(ejml2.matrix))
    case _:Tensor1 => val result = this.copy; t.foreachActiveElement((i,v) => result+=(i,v)); result
  }
  override def blankCopy: Tensor1 = new EJMLTensor1(dim1)
  override def copy: Tensor1 = new EJMLTensor1(matrix.copy())
  override def sum: Double = matrix.elementSum()
  override def :=(ds: DoubleSeq): Unit = ds match {
    case ejml: EJMLTensor1 => matrix.set(ejml.matrix)
    case _ => super.:=(ds)
  }
  override def outer(t: Tensor): Tensor = t match {
    case t:EJMLTensor1 => val A = new EJMLTensor2(dim1,t.dim1); VectorVectorMult.outerProd(matrix.getMatrix,t.matrix.getMatrix,A.matrix.getMatrix); A
    case _ => super.outer(t)
  }
  override def dot(ds: DoubleSeq): Double = ds match {
    case ejml2:EJMLTensor1 => matrix.dot(ejml2.matrix)
    case _ => var sum = 0.0; ds.foreachActiveElement((i,v) => sum+=matrix.get(i)*v);sum
  }
}

class EJMLTensor2(override val matrix:SimpleMatrix) extends EJMLTensor with Tensor2 {
  def this(dim1:Int,dim2:Int) = this(new SimpleMatrix(dim1,dim2))
  override def dim1: Int = matrix.numRows()
  override def activeDomain2: IntSeq = new RangeIntSeq(0,dim2)
  override def activeDomain1: IntSeq = new RangeIntSeq(0,dim1)
  override def dim2: Int = matrix.numCols()
  override def activeDomainSize: Int = dim1*dim2
  override def activeDomain: IntSeq = new RangeIntSeq(0,dim1*dim2)
  override def copy: Tensor2 = new EJMLTensor2(matrix.copy())
  override def blankCopy: Tensor2 = new EJMLTensor2(dim1,dim2)
  override def +=(i: Int, j: Int, v: Double): Unit = matrix.set(i,j, apply(i,j) + v)
  override def *(t: Tensor1): Tensor1 = t match {
    case ejml1:EJMLTensor1 => new EJMLTensor1(matrix.mult(ejml1.matrix))
    case _ => val res = new EJMLTensor1(dim1);  for(i <- 0 until dim1) { var sum = 0.0; t.foreachActiveElement((j,v) => sum+= apply(i,j)*v); res.update(i,sum)}; res
  }
  override def leftMultiply(t: Tensor1): Tensor1 = t match {
    case ejml1:EJMLTensor1 => new EJMLTensor1(matrix.transpose().mult(ejml1.matrix))
    case _ => val res = new EJMLTensor1(dim2);  for(j <- 0 until dim2) { var sum = 0.0; t.foreachActiveElement((i,v) => sum+= apply(i,j)*v); res.update(j,sum)}; res
  }
  override def diag: Tensor1 = new EJMLTensor1(matrix.extractDiag())
  override def apply(i: Int, j: Int): Double = matrix.get(i, j)
  override def sum: Double = matrix.elementSum()
  override def :=(ds: DoubleSeq): Unit = ds match {
    case ejml: EJMLTensor2 => matrix.getMatrix.set(ejml.matrix.getMatrix)
    case _ => super.:=(ds)
  }
  override def dot(ds: DoubleSeq): Double = ds match {
    case ejml2:EJMLTensor2 => matrix.elementMult(ejml2.matrix).elementSum()
    case _ => var sum = 0.0; ds.foreachActiveElement((i,v) => sum+=matrix.get(i)*v);sum
  }
}





