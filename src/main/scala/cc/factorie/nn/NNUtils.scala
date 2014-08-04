package cc.factorie.nn

import cc.factorie.la._


object NNUtils  extends TensorImplementation {
  //this setting defines the implementation for matrices and vectors; EJML/NATIVE is best for small matrix computations and JBLAS for bigger ones
  private var tensorImpl:TensorImplementation = NATIVE
  def setTensorImplementation(impl:TensorImplementation) = tensorImpl = impl


  object NATIVE extends TensorImplementation {
    override def newDense(dim1: Int): Tensor1 = new DenseTensor1(dim1)
    override def newDense(dim1: Int, dim2: Int): Tensor2 = new DenseTensor2(dim1,dim2)
    override def newDense(dim1: Int, dim2: Int, dim3: Int): Tensor3 = new DenseTensor3(dim1,dim2,dim3)
    override def newDense(dim1: Int, dim2: Int, dim3: Int, dim4: Int): Tensor4 = new DenseTensor4(dim1,dim2,dim3,dim4)
  }

  object JBLAS extends TensorImplementation {
    override def newDense(dim1: Int): Tensor1 = new JBlasTensor1(dim1)
    override def newDense(dim1: Int, dim2: Int): Tensor2 = new JBlasTensor2(dim1,dim2)
    override def newDense(dim1: Int, dim2: Int, dim3: Int): Tensor3 = throw new NotImplementedError() //Not Implemented yet, store an array of JBlasTensor2
    override def newDense(dim1: Int, dim2: Int, dim3: Int, dim4: Int): Tensor4 = throw new NotImplementedError() //Not Implemented yet, store Array of JBLASTensor3
  }
  object EJML extends TensorImplementation {
    override def newDense(dim1: Int): Tensor1 = new EJMLTensor1(dim1)
    override def newDense(dim1: Int, dim2: Int): Tensor2 = new EJMLTensor2(dim1,dim2)
    override def newDense(dim1: Int, dim2: Int, dim3: Int): Tensor3 = throw new NotImplementedError() //Not Implemented yet, store an array of JBlasTensor2
    override def newDense(dim1: Int, dim2: Int, dim3: Int, dim4: Int): Tensor4 = throw new NotImplementedError() //Not Implemented yet, store Array of JBLASTensor3
  }

/* simple performance tests:
  big matrix multiplications + addition: winner JBLAS
  small matrix multiplication + addition: winner EJML
  big elementwise matrix multiplication: winner EJML
  small elementwise matrix multiplication: winner EJML
*/
  def main (args: Array[String]) {
    import scala.util.Random

    val dim = 25
    
    val dense1 = NATIVE.fillDense(dim)((_)=>Random.nextDouble())
    val dense2 = NATIVE.fillDense(dim,dim)((_,_)=>Random.nextDouble())

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
  def newDense(dim1:Int):Tensor1
  def newDense(dim1:Int,dim2:Int):Tensor2
  def newDense(dim1:Int,dim2:Int,dim3:Int):Tensor3
  def newDense(dim1:Int,dim2:Int,dim3:Int,dim4:Int):Tensor4

  def fillDense(dim1:Int)(init: (Int) => Double):Tensor1 = {
    val t = newDense(dim1)
    (0 until dim1).foreach(i => t.+=(i,init(i)))
    t
  }
  def fillDense(dim1:Int,dim2:Int)(init: (Int,Int) => Double):Tensor2 = {
    val t = newDense(dim1,dim2)
    for(i <- 0 until dim1; j <- 0 until dim2) t.+=(i,j,init(i,j))
    t
  }
  def fillDense(dim1:Int,dim2:Int,dim3:Int)(init: (Int,Int,Int) => Double):Tensor3 = {
    val t = newDense(dim1,dim2,dim3)
    for(i <- 0 until dim1; j <- 0 until dim2; k<- 0 until dim3) t.+=(i,j,k,init(i,j,k))
    t
  }
  def fillDense(dim1:Int,dim2:Int,dim3:Int,dim4:Int)(init: (Int,Int,Int,Int) => Double):Tensor4 = {
    val t = newDense(dim1,dim2,dim3,dim4)
    for(i <- 0 until dim1; j <- 0 until dim2; k<- 0 until dim3; l <- 0 until dim4) t.+=(i,j,k,l,init(i,j,k,l))
    t
  }

}