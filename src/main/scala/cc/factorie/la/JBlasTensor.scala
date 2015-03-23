package cc.factorie.la

import cc.factorie.util._
import org.ejml.ops.CommonOps
import org.jblas.DoubleMatrix

trait JBlasTensor extends Tensor with DenseTensor {
  def jblas:DoubleMatrix
  override protected def _initialArray: Array[Double] = jblas.data
  override def forallActiveElements(f: (Int, Double) => Boolean): Boolean = {
    var i = 0
    var res = true
    while(i < jblas.length && res) {
      res &&= f(i, apply(i))
      i+=1
    }
    res
  }
  override def zero(): Unit = jblas.muli(0.0)
  override def isDense: Boolean = true
  override def dot(ds: DoubleSeq): Double = ds match {
    case jblas2:JBlasTensor => jblas.dot(jblas2.jblas)
    case _ =>
      var sum = 0.0
      ds.foreachActiveElement((i,v) => sum += v * apply(i))
      sum
  }
  override def *=(ds: DoubleSeq): Unit = ds match {
    case jblas2:JBlasTensor => jblas.muli(jblas2.jblas)
    case _ => ds.foreachActiveElement((i,v) => update(i,v * apply(i)))
  }
  override def +=(ds: DoubleSeq, factor: Double): Unit = ds match {
    case jblas2:JBlasTensor => if(factor==1.0) jblas.addi(jblas2.jblas) else jblas.addi(jblas2.jblas.mul(factor))
    case _ => ds.foreachActiveElement((i,v) => { +=(i,v*factor) })
  }
  override def oneNorm: Double = jblas.norm1()
  override def twoNormSquared: Double = { val d = jblas.norm2(); d*d}
  override def *=(d: Double): Unit = jblas.muli(d)
  override def +=(d: Double): Unit = jblas.addi(d)
  override def :=(ds: DoubleSeq): Unit = ds match {
    case t: Tensor => zero(); +=(t)
    case _ => super.:=(ds)
  }
}

//Use JBLAS if your tensors are very big
//Use EJML for small matrices
//Try not to mix different kinds of implementations... Use NNUtils.newDense-methods
class JBlasTensor1(override val jblas:DoubleMatrix) extends JBlasTensor with DenseTensorLike1 {
  def this(array:Array[Double]) = this(new DoubleMatrix(array))
  def this(dim:Int) = this(new DoubleMatrix(dim))
  def this(dim:Int, f:Int=>Double) = this({
    val t = new DoubleMatrix(dim)
    (0 until dim).foreach(i => t.put(i,f(i)))
    t
  })
  override def dim1: Int = jblas.length
  override def update(i: Int, v: Double): Unit = jblas.put(i,v)

  override def blankCopy: Tensor1 = new JBlasTensor1(dim1)
  override def copy: Tensor1 = {
    val r = new JBlasTensor1(dim1)
    r.jblas.copy(jblas)
    r
  }
  override def +=(i: Int, incr: Double): Unit = jblas.put(i,apply(i)+incr)
  override def apply(i: Int): Double = jblas.get(i)
  override def outer(t: Tensor): Tensor = t match {
    case t:Tensor1 =>
      val result = new JBlasTensor2(dim1,t.dim1)
      t.foreachActiveElement((j,v1) => result.jblas.putColumn(j,jblas.mul(v1)))
      result
    case _ => super.outer(t)
  }
}

//jblas counts row first, so it is a little different from factorie which counts columns first
class JBlasTensor2(val jblas:DoubleMatrix) extends JBlasTensor with DenseTensorLike2 {
  private def tfIdx(i:Int) = ((i/dim2)%dim1) + ((i%dim2)*dim1)
  def this(dim1:Int,dim2:Int, f:(Int,Int)=>Double) = this({
    val t = new DoubleMatrix(dim1,dim2)
    (0 until dim1).foreach(i => (0 until dim2).foreach(j => t.put(i,j,f(i,j))))
    t
  })
  def this(dim1:Int,dim2:Int) = this(new DoubleMatrix(dim1,dim2))
  override def dim1: Int = jblas.rows
  override def dim2: Int = jblas.columns
  override def apply(i: Int): Double = jblas.get(tfIdx(i))
  override def update(i: Int, v: Double): Unit = jblas.put(tfIdx(i),v)
  override def forallActiveElements(f: (Int, Double) => Boolean): Boolean = {
    var i = 0
    var res = true
    while(i < jblas.length && res) {
      res &&= f(i, apply(i))
      i+=1
    }
    res
  }
  override def +=(i: Int, incr: Double): Unit = jblas.put(tfIdx(i),incr+apply(i))
  override def leftMultiply(t: Tensor1): Tensor1 = {
    val res = new JBlasTensor1(dim2)
    leftMultiply(t,res)
    res
  }
  override def leftMultiply(t: Tensor1, result: Tensor1): Unit = (t,result) match  {
    case (t:JBlasTensor1,res:JBlasTensor1) =>
      jblas.transpose().mmuli(t.jblas,res.jblas)
    case _ =>
      var j = 0
      while (j < dim2) {
        var sum = 0.0
        t.foreachActiveElement((i,v) => sum+= apply(i,j)*v)
        result.update(j,sum)
        j+=1
      }
  }
  override def apply(i: Int, j: Int): Double = jblas.get(i,j)
  override def *(t: Tensor1): Tensor1 = {
    val res = new JBlasTensor1(dim1)
    *(t,res)
    res
  }
  override def *(t: Tensor1, res:Tensor1): Unit = (t,res) match {
    case (t:JBlasTensor1,res:JBlasTensor1) =>
      jblas.mmuli(t.jblas, res.jblas)
    case _ =>
      var i = 0
      while (i < dim1) {
        var sum = 0.0
        t.foreachActiveElement((j, v) => sum += apply(i, j) * v)
        res.update(i, sum)
        i+=1
      }
  }
  override def blankCopy: Tensor2 = new JBlasTensor2(dim1,dim2)
  override def copy: Tensor2 = {
    val r = new JBlasTensor2(dim1,dim2)
    r.jblas.copy(jblas)
    r
  }
}

