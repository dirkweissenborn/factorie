package cc.factorie.nn

import cc.factorie.la._
import org.ejml.data.DenseMatrix64F
import org.ejml.simple.SimpleMatrix
import org.jblas.DoubleMatrix

import scala.util.Random

object TensorUtils extends TensorImplementation {
  //this setting defines the implementation for matrices and vectors; EJML is best for small matrix
  // computations and JBLAS for bigger ones; FACTORIE is usually also usually slower
  private var tensorImpl:TensorImplementation = FACTORIE
  def setTensorImplementation(impl:TensorImplementation) = tensorImpl = impl
  
  type T1 = Tensor1
  type T2 = Tensor2
  type T3 = Tensor3
  type T4 = Tensor4
  override def convert(t: Tensor1): T1 = tensorImpl.convert(t)
  override def convert(t: Tensor2): T2 = tensorImpl.convert(t)
  override def convert(t: Tensor3): T3 = tensorImpl.convert(t)
  override def convert(t: Tensor4): T4 = tensorImpl.convert(t)

  def randomTensor1(dim:Int,scale:Double = 1.0)(implicit rng:Random) = TensorUtils.fillDense(dim)(_ => rng.nextDouble()*scale)

  object FACTORIE extends TensorImplementation {    
    override type T1 = DenseTensor1
    override type T2 = DenseTensor2
    override type T3 = DenseTensor3
    override type T4 = DenseTensor4
    override def newDense(dim1: Int): T1 = new DenseTensor1(dim1)
    override def newDense(dim1: Int, dim2: Int): T2 = new DenseTensor2(dim1,dim2)
    override def newDense(dim1: Int, dim2: Int, dim3: Int): T3 = new DenseTensor3(dim1,dim2,dim3)
    override def newDense(dim1: Int, dim2: Int, dim3: Int, dim4: Int): T4 = new DenseTensor4(dim1,dim2,dim3,dim4)
    override def convert(t: Tensor1): T1 = new DenseTensor1(t.asArray)
    override def convert(t: Tensor2): T2 = new DenseTensor2(t.dim1,t.dim2) {
      override protected def _initialArray: Array[Double] = t.asArray
    }
    override def convert(t: Tensor3): T3 = new DenseTensor3(t.dim1,t.dim2,t.dim3) {
      override protected def _initialArray: Array[Double] = t.asArray
    }
    override def convert(t: Tensor4): T4 = new DenseTensor4(t.dim1,t.dim2,t.dim3,t.dim4) {
      override protected def _initialArray: Array[Double] = t.asArray
    }
  }
  object JBLAS extends TensorImplementation {
    override type T1 = JBlasTensor1
    override type T2 = JBlasTensor2
    override type T3 = FixedLayers1DenseTensor3[JBlasTensor2]
    override type T4 = Tensor4
    override def newDense(dim1: Int): T1 = new JBlasTensor1(dim1)
    override def newDense(dim1: Int, dim2: Int): T2 = new JBlasTensor2(dim1,dim2)
    override def newDense(dim1: Int, dim2: Int, dim3: Int): T3 = new FixedLayers1DenseTensor3(Array.fill(dim3)(new 
        JBlasTensor2(dim1,dim2))) //Not Implemented yet, store an array of JBlasTensor2
    override def newDense(dim1: Int, dim2: Int, dim3: Int, dim4: Int): T4 = throw new NotImplementedError() //Not
    // Implemented yet, store Array of JBLASTensor3
    override def convert(t: Tensor1): T1 = new JBlasTensor1(new DoubleMatrix(t.asArray))
    override def convert(t: Tensor2): T2 = new JBlasTensor2(new DoubleMatrix(t.asArray).reshape(t.dim1,t.dim2))
    override def convert(t: Tensor3): T3 = throw new NotImplementedError()
    override def convert(t: Tensor4): T4 = throw new NotImplementedError()
  }
  object EJML extends TensorImplementation {
    override type T1 = EJMLTensor1
    override type T2 = EJMLTensor2
    override type T3 = FixedLayers1DenseTensor3[EJMLTensor2]
    override type T4 = Tensor4
    override def newDense(dim1: Int): T1 = new EJMLTensor1(dim1)
    override def newDense(dim1: Int, dim2: Int): T2 = new EJMLTensor2(dim1,dim2)
    override def newDense(dim1: Int, dim2: Int, dim3: Int): T3 = 
      new FixedLayers1DenseTensor3(Array.fill(dim3)(new EJMLTensor2(dim1,dim2))) 
    override def newDense(dim1: Int, dim2: Int, dim3: Int, dim4: Int): T4 = throw new NotImplementedError() //Not 
    // Implemented yet, store Array of EJMLTensor3
    override def convert(t:Tensor1): T1 = 
      new EJMLTensor1(new SimpleMatrix(DenseMatrix64F.wrap(t.dim1,1,t.asArray)))
    override def convert(t: Tensor2): T2 =
      new EJMLTensor2(new SimpleMatrix(DenseMatrix64F.wrap(t.dim1,t.dim2,t.asArray)))
    override def convert(t: Tensor3): T3 = throw new NotImplementedError()
    override def convert(t: Tensor4): T4 = throw new NotImplementedError()
  }

  def concatenateTensor1(t1:Tensor1,t2:Tensor1):Tensor1 = (t1,t2) match {
    case (t1:JBlasTensor1,t2:JBlasTensor1) =>
      new JBlasTensor1(DoubleMatrix.concatVertically(t1.jblas,t2.jblas))
    case (t1:EJMLTensor1,t2:EJMLTensor1) =>
      new EJMLTensor1(t1.ejml.combine(t1.dim1,0,t2.ejml))
    case _ => new ConcatenatedTensor(Seq(t1,t2))
  }


  //Usually used with a prior concatenateTensor1
  def splitTensor1(t:Tensor1, at:Int) = t match {
    case t:JBlasTensor1 =>
      (new JBlasTensor1(t.jblas.getRowRange(0,at,0)),new JBlasTensor1(t.jblas.getRowRange(at,t.dim1,0)))
    case t:EJMLTensor1 =>
      (new EJMLTensor1(t.ejml.extractMatrix(0,at,0,1)),new EJMLTensor1(t.ejml.extractMatrix(at,t.dim1,0,1)))
    case t:ConcatenatedTensor =>
      if(t.tensors.length == 2 && t.tensors.head.length == at)
        (t.tensors(0).asInstanceOf[Tensor1], t.tensors(1).asInstanceOf[Tensor1])
      else
        throw new Error(s"Splitting ConcatenatedTensor with more than 2 tensors or where first tensors length does not is not the splitting point is not supported")
    case _ => throw new Error(s"Splitting ${t.getClass.getSimpleName} not supported")
  }
  
  def main (args: Array[String]) {
    import scala.util.Random

    val dim = 25
    
    val dense1 = FACTORIE.fillDense(dim)((_)=>Random.nextDouble())
    val dense2 = FACTORIE.fillDense(dim,dim)((_,_)=>Random.nextDouble())

    val jblas1 = JBLAS.fillDense(dim)((_)=>Random.nextDouble())
    val jblas2 = JBLAS.fillDense(dim,dim)((_,_)=>Random.nextDouble())

    val ejml1 = EJML.fillDense(dim)((_)=>Random.nextDouble())
    val ejml2 = EJML.fillDense(dim,dim)((_,_)=>Random.nextDouble())

  /*
    val mtj1 = new matrix.DenseMatrix(dim,1)
    (0 until dim).foreach(i => mtj1.set(i,0, Random.nextDouble()))
    val mtj2 = new matrix.DenseMatrix(dim,dim)
    for (i <- 0 until dim;j <- 0 until dim) mtj2.set(i,j, Random.nextDouble())

    val mtj3 = mtj1.copy()

    val breeze1 = DenseMatrix.rand(dim,1)
    val breeze2 = DenseMatrix.rand(dim,dim)

    val t = dense2 * dense1  + dense1
    val t1 = breeze2 * breeze1 + breeze1
    val t2 = jblas2 * jblas1  + jblas1
    val t3 = ejml2.mult(ejml1).plus(ejml1)
    val t4 = mtj2.mult(mtj1,mtj3)
*/

    var start =System.currentTimeMillis()
    var i = 0
    while(i <250000/dim) { dense2 * dense1  + dense1;i+=1}
    println(System.currentTimeMillis()-start)

    start =System.currentTimeMillis()
    i = 0
    while(i <250000/dim) { jblas2 * jblas1  + jblas1;i+=1}
    println(System.currentTimeMillis()-start)

    start =System.currentTimeMillis()
    i = 0
    while(i <250000/dim) { ejml2 * ejml1  + ejml1;i+=1}
    println(System.currentTimeMillis()-start)
    //########################################
    i = 0
    while(i <2500000/dim) { dense1.copy*=dense1;i+=1}
    println(System.currentTimeMillis()-start)

    start =System.currentTimeMillis()
    i = 0
    while(i <2500000/dim) { jblas1.copy.*=(jblas1);i+=1}
    println(System.currentTimeMillis()-start)

    start =System.currentTimeMillis()
    i = 0
    while(i <2500000/dim) { ejml1.copy.*=(ejml1);i+=1}
    println(System.currentTimeMillis()-start)
  }

  override def newDense(dim1: Int): Tensor1 = tensorImpl.newDense(dim1)
  override def newDense(dim1: Int, dim2: Int): Tensor2 = tensorImpl.newDense(dim1,dim2)
  override def newDense(dim1: Int, dim2: Int, dim3: Int): Tensor3 = tensorImpl.newDense(dim1,dim2,dim3)
  override def newDense(dim1: Int, dim2: Int, dim3: Int, dim4: Int): Tensor4 = tensorImpl.newDense(dim1,dim2,dim3,dim4)
  def newDense(t:Tensor):Tensor = t match {
    case t1:Tensor1 => newDense(t1.dim1)
    case t2:Tensor2 => newDense(t2.dim1,t2.dim2)
    case t3:Tensor3 => newDense(t3.dim1,t3.dim2,t3.dim3)
    case t4:Tensor4 => newDense(t4.dim1,t4.dim2,t4.dim3,t4.dim4)
  }
}

trait TensorImplementation {
  type T1 <: Tensor1
  type T2 <: Tensor2
  type T3 <: Tensor3
  type T4 <: Tensor4
  
  def newDense(dim1:Int):T1
  def newDense(dim1:Int,dim2:Int):T2
  def newDense(dim1:Int,dim2:Int,dim3:Int):T3
  def newDense(dim1:Int,dim2:Int,dim3:Int,dim4:Int):T4

  def fillDense(dim1:Int)(init: (Int) => Double):T1 = {
    val t = newDense(dim1)
    (0 until dim1).foreach(i => t.+=(i,init(i)))
    t
  }
  def fillDense(dim1:Int,dim2:Int)(init: (Int,Int) => Double):T2 = {
    val t = newDense(dim1,dim2)
    for(i <- 0 until dim1; j <- 0 until dim2) t.+=(i,j,init(i,j))
    t
  }
  def fillDense(dim1:Int,dim2:Int,dim3:Int)(init: (Int,Int,Int) => Double):T3 = {
    val t = newDense(dim1,dim2,dim3)
    for(i <- 0 until dim1; j <- 0 until dim2; k<- 0 until dim3) t.+=(i,j,k,init(i,j,k))
    t
  }
  def fillDense(dim1:Int,dim2:Int,dim3:Int,dim4:Int)(init: (Int,Int,Int,Int) => Double):T4 = {
    val t = newDense(dim1,dim2,dim3,dim4)
    for(i <- 0 until dim1; j <- 0 until dim2; k<- 0 until dim3; l <- 0 until dim4) t.+=(i,j,k,l,init(i,j,k,l))
    t
  }
  
  def convert(t:Tensor1):T1
  def convert(t:Tensor2):T2
  def convert(t:Tensor3):T3
  def convert(t:Tensor4):T4

}