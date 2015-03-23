package cc.factorie.la

import cc.factorie.util.{DenseDoubleSeq, DoubleSeq, IntSeq, RangeIntSeq}
import org.ejml.alg.dense.mult.VectorVectorMult
import org.ejml.ops.{NormOps, CommonOps}
import org.ejml.simple.SimpleMatrix

trait EJMLTensor extends Tensor with DenseTensor {
  val ejml:SimpleMatrix
  override protected def _initialArray: Array[Double] = ejml.getMatrix.getData
  override def update(i: Int, v: Double): Unit = ejml.set(i,v)
  override def isDense: Boolean = true
  override def zero(): Unit = ejml.zero()
  override def apply(i: Int): Double = ejml.get(i)
  override def *=(d: Double): Unit = CommonOps.scale(d,ejml.getMatrix)
  override def oneNorm: Double = NormOps.fastNormP(ejml.getMatrix,1.0)
  override def twoNormSquared: Double = { val f = ejml.normF(); f*f }
  override def +=(i: Int, incr: Double): Unit = ejml.set(i, apply(i) + incr)
  override def *=(i: Int, incr: Double): Unit = ejml.set(i, apply(i) * incr)
  override def +=(ds: DoubleSeq, factor: Double): Unit = ds match {
    case ejml2:EJMLTensor =>
      if(factor == 1.0) CommonOps.addEquals(ejml.getMatrix, ejml2.ejml.getMatrix)
      else CommonOps.addEquals(ejml.getMatrix,factor,ejml2.ejml.getMatrix)
    case _ => ds.foreachActiveElement((i,v) => +=(i,v))
  }
  override def :=(ds: DoubleSeq): Unit = ds match {
    case t: Tensor => zero(); +=(t)
    case _ => super.:=(ds)
  }
  override def /=(ds: DoubleSeq): Unit = ds match {
    case ejml2:EJMLTensor =>
      CommonOps.elementDiv(ejml.getMatrix, ejml2.ejml.getMatrix)
    case _ => super./=(ds)
  }
  override def *=(ds: DoubleSeq): Unit = ds match {
    case ejml2:EJMLTensor => CommonOps.elementMult(ejml.getMatrix, ejml2.ejml.getMatrix)
    case _ => ds.foreachActiveElement((i,v) => *=(i,v))
  }
  override def +=(d: Double): Unit = CommonOps.add(ejml.getMatrix,d)
  override def forallActiveElements(f: (Int, Double) => Boolean): Boolean = {
    val a = ejml.getMatrix.iterator(true, 0, 0, ejml.numRows() - 1, ejml.numCols() - 1)
    var res = true
    while (res && a.hasNext) {
      val n = a.next()
      res &&= f(a.getIndex, n)
    }
    res
  }
}

class EJMLTensor1(override val ejml:SimpleMatrix) extends EJMLTensor with DenseTensorLike1 {
  assert(ejml.isVector)
  def this(dim1:Int) = this(new SimpleMatrix(dim1,1))
  override def dim1: Int = ejml.numRows()
  override def *(f: Double): Tensor1 = new EJMLTensor1(ejml.scale(f))
  override def +(t: Tensor1): Tensor1 = t match {
    case ejml2:EJMLTensor1 => new EJMLTensor1(ejml.plus(ejml2.ejml))
    case _:Tensor1 => val result = this.copy; t.foreachActiveElement((i,v) => result+=(i,v)); result
  }
  override def blankCopy: Tensor1 = new EJMLTensor1(dim1)
  override def copy: Tensor1 = new EJMLTensor1(ejml.copy())
  override def sum: Double = ejml.elementSum()
  override def outer(t: Tensor): Tensor = t match {
    case t:EJMLTensor1 => 
      val A = new EJMLTensor2(dim1,t.dim1)
      VectorVectorMult.outerProd(ejml.getMatrix,t.ejml.getMatrix,A.ejml.getMatrix); A
    case _ => super.outer(t)
  }
  override def dot(ds: DoubleSeq): Double = ds match {
    case ejml2:EJMLTensor1 => ejml.dot(ejml2.ejml)
    case _ => var sum = 0.0; ds.foreachActiveElement((i,v) => sum+=ejml.get(i)*v);sum
  }
}

class EJMLTensor2(override val ejml:SimpleMatrix) extends EJMLTensor with DenseTensorLike2 {
  def this(dim1:Int,dim2:Int) = this(new SimpleMatrix(dim1,dim2))
  override def dim1: Int = ejml.numRows()
  override def dim2: Int = ejml.numCols()
  override def copy: Tensor2 = new EJMLTensor2(ejml.copy())
  override def blankCopy: Tensor2 = new EJMLTensor2(dim1,dim2)
  override def +=(i: Int, j: Int, v: Double): Unit = ejml.set(i,j, apply(i,j) + v)
  override def *(t: Tensor1): Tensor1 = {
    val res = new EJMLTensor1(dim1)
    *(t,res)
    res
  }
  override def *(t: Tensor1, result:Tensor1): Unit = (t,result) match {
    case (t: EJMLTensor1, result: EJMLTensor1) =>
      CommonOps.mult(ejml.getMatrix, t.ejml.getMatrix, result.ejml.getMatrix)
    case _ =>
      var i = 0
      while (i < dim1) {
        var sum = 0.0
        t.foreachActiveElement((j, v) => sum += apply(i, j) * v)
        result.update(i, sum)
        i+=1
      }
  }
  override def leftMultiply(t: Tensor1): Tensor1 = {
    val res = new EJMLTensor1(dim2)
    leftMultiply(t,res)
    res
  }
  override def leftMultiply(t: Tensor1, result: Tensor1): Unit = (t,result) match  {
    case (t:EJMLTensor1,res:EJMLTensor1) =>
      CommonOps.mult(ejml.transpose().getMatrix, t.ejml.getMatrix, res.ejml.getMatrix)
    case _ =>
      var j = 0
      while (j < dim2) {
        var sum = 0.0
        t.foreachActiveElement((i,v) => sum+= apply(i,j)*v)
        result.update(j,sum)
        j+=1
      }
  }
  override def diag: Tensor1 = new EJMLTensor1(ejml.extractDiag())
  override def apply(i: Int, j: Int): Double = ejml.get(i, j)
  override def sum: Double = ejml.elementSum()
  override def dot(ds: DoubleSeq): Double = ds match {
    case t:EJMLTensor2 => ejml.elementMult(t.ejml).elementSum()
    case _ => var sum = 0.0; ds.foreachActiveElement((i,v) => sum+=ejml.get(i)*v);sum
  }
}





