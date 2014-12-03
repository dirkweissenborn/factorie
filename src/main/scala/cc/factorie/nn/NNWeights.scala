package cc.factorie.nn

import cc.factorie.la._
import cc.factorie.model._

trait NNWeights {
  type ConnectionType <: Connection
  type FamilyType <: NNWeights
  def weights:Weights
  def numVariables:Int

  protected def _outputLayers(f: FamilyType#ConnectionType): Seq[NNLayer]
  protected def _inputLayers(f: FamilyType#ConnectionType): Seq[NNLayer]
  //increments input of output layers (by calling l.incrementInput(inc:Tensor1) for each output layer l)
  protected def _forwardPropagate(f: FamilyType#ConnectionType): Unit
  protected def _backPropagate(f: FamilyType#ConnectionType): Unit
  //returns objective on weights and propagates objective to input layers from given objective on output layer
  protected def _backPropagateGradient(f: FamilyType#ConnectionType): Tensor

  trait Connection {
    def weights = family.weights.value
    def family:FamilyType = NNWeights.this.asInstanceOf[FamilyType]
    val outputLayers = NNWeights.this._outputLayers(this.asInstanceOf[FamilyType#ConnectionType])
    val inputLayers = NNWeights.this._inputLayers(this.asInstanceOf[FamilyType#ConnectionType])
    def forwardPropagate() = NNWeights.this._forwardPropagate(this.asInstanceOf[FamilyType#ConnectionType])
    def backPropagateGradient = NNWeights.this._backPropagateGradient(this.asInstanceOf[FamilyType#ConnectionType])
    def backPropagate() = NNWeights.this._backPropagate(this.asInstanceOf[FamilyType#ConnectionType])
    def numVariables = NNWeights.this.numVariables
  }
  def newConnection(layers:Seq[NNLayer]):Connection
}

trait NNWeights1[N1<:NNLayer] extends NNWeights {
  type ConnectionType <: Connection
  case class Connection(_1:N1) extends super.Connection
  override val numVariables: Int = 1
  def newConnection(_1:N1) = new Connection(_1)
  override def newConnection(layers: Seq[NNLayer]): Connection = {
    assert(layers.size == numVariables, s"Cannot create connection between ${layers.size} layers using ${getClass.getName}")
    newConnection(layers.head.asInstanceOf[N1])
  }
}
trait NNWeights2[N1<:NNLayer,N2<:NNLayer] extends NNWeights  {
  type ConnectionType <: Connection
  case class Connection(_1:N1,_2:N2) extends super.Connection
  override val numVariables: Int = 2
  def newConnection(_1:N1,_2:N2) = new Connection(_1,_2)
  override def newConnection(layers: Seq[NNLayer]): Connection = {
    assert(layers.size == numVariables, s"Cannot create connection between ${layers.size} layers using ${getClass.getName}")
    newConnection(layers.head.asInstanceOf[N1],layers.last.asInstanceOf[N2])
  }
}
trait NNWeights3[N1<:NNLayer,N2<:NNLayer,N3<:NNLayer] extends NNWeights  {
  type ConnectionType <: Connection
  type FamilyType <: NNWeights3[N1,N2,N3]
  case class Connection(_1:N1, _2:N2, _3:N3) extends super.Connection
  override val numVariables: Int = 3
  def newConnection(_1:N1,_2:N2,_3:N3) = new Connection(_1,_2,_3)
  override def newConnection(layers: Seq[NNLayer]): Connection = {
    assert(layers.size == numVariables, s"Cannot create connection between ${layers.size} layers using ${getClass.getName}")
    newConnection(layers.head.asInstanceOf[N1],layers(1).asInstanceOf[N2],layers.last.asInstanceOf[N3])  }
}
trait NNWeights4[N1<:NNLayer,N2<:NNLayer,N3<:NNLayer,N4<:NNLayer] extends NNWeights {
  type ConnectionType <: Connection
  case class Connection(_1:N1, _2:N2, _3:N3, _4:N4) extends super.Connection
  override val numVariables: Int = 4
  def newConnection(_1:N1,_2:N2,_3:N3,_4:N4) = new Connection(_1,_2,_3,_4)
  override def newConnection(layers: Seq[NNLayer]): Connection = {
    assert(layers.size == numVariables, s"Cannot create connection between ${layers.size} layers using ${getClass.getName}")
    newConnection(layers.head.asInstanceOf[N1],layers(1).asInstanceOf[N2],layers(2).asInstanceOf[N3],layers.last.asInstanceOf[N4])
  }
}

//The following classes can be sub-classed in your neural network model
trait Bias[N1<:NNLayer] extends NNWeights1[N1] {
  override type FamilyType <: Bias[N1]
  override type ConnectionType = Connection
  override def weights: Weights1
  override protected def _outputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._1)
  override protected def _inputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq[NNLayer]()
  override protected def _backPropagateGradient(f: FamilyType#ConnectionType): Tensor = weights.value match {
    case w: Tensor1 => f._1.objectiveGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw objective because this should not be possible
  }
  override protected def _forwardPropagate(f: FamilyType#ConnectionType): Unit = f._1.incrementInput(weights.value)
  override protected def _backPropagate(f: FamilyType#ConnectionType): Unit = {}
}
//_1 is input and _2 is output of these weights, if these weights are used as NeuralNetworkWeights
trait BasicLayerToLayerWeights[N1 <: NNLayer, N2 <:NNLayer] extends NNWeights2[N1,N2] {
  override type FamilyType <: BasicLayerToLayerWeights[N1,N2]
  override type ConnectionType = Connection
  override def weights: Weights2
  override protected def _outputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._2)
  override protected def _inputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._1)
  override protected def _forwardPropagate(f: FamilyType#ConnectionType): Unit = weights.value match {
    case weights: Tensor2 => f._2.incrementInput(f._1.value * weights)
  }
  override protected def _backPropagateGradient(f: FamilyType#ConnectionType): Tensor2 = weights.value match {
    case w: Tensor2 =>
      val outGradient = f._2.objectiveGradient
      f._1.incrementObjectiveGradient(w * outGradient)
      (f._1.value outer outGradient).asInstanceOf[Tensor2]
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw objective because this should not be possible
  }

  override protected def _backPropagate(f: FamilyType#ConnectionType): Unit = {}
}
//This can be useful if you do not want to precalculate something as input for different layers
object IdentityWeights extends NNWeights2[NNLayer,NNLayer] {
  override type ConnectionType = Connection
  override type FamilyType = NNWeights2[NNLayer,NNLayer]
  override val weights: Weights2 = null //no weights here
  override protected def _outputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._2)
  override protected def _inputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._1)
  override protected def _forwardPropagate(f: FamilyType#ConnectionType): Unit = f._2.incrementInput(f._1.value)
  override protected def _backPropagateGradient(f: FamilyType#ConnectionType): Tensor2 = {
      val outGradient = f._2.objectiveGradient
      f._1.incrementObjectiveGradient(outGradient)
      null //Hack, just return an empty tensor because we do not want to update these weights
  }

  override protected def _backPropagate(f: FamilyType#ConnectionType): Unit = {}
}
//_1 and _2 are input layers and _3 is output layer if these weights are considered NeuralNetworkWeights, (e.g., see RecurrentNeuralTensorNetworks by Socher et al.)
trait NeuralTensorWeights[N1<:NNLayer,N2<:NNLayer,N3<:NNLayer] extends NNWeights3[N1,N2,N3] {
  override type FamilyType = NeuralTensorWeights[N1,N2,N3]
  override type ConnectionType = Connection
  //dimensionality: dim1 x dim2 x dim3
  override def weights: Weights3
  override protected def _outputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._3)
  override protected def _inputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._1, f._2)
  override protected def _forwardPropagate(f: FamilyType#ConnectionType): Unit = weights.value match {
    case w:FixedLayers1DenseTensor3 =>
      val input = NNUtils.fillDense(f._3.value.length)(k => (w.matrices(k) * f._2.value) dot f._1.value)
      f._3.incrementInput(input)
    case w: Tensor3 =>
      val input = NNUtils.newDense(f._3.value.length)
      for (k <- 0 until input.dim1)
        for (i <- 0 until f._1.value.dim1)
          for (j <- 0 until f._2.value.dim1) {
            val v1 = f._1.value(i)
            val v2 =f._2.value(j)
            val weight: Double = weights.value(i, j, k)
            input +=(k, weight * v1 * v2)
          }
      f._3.incrementInput(input)
  }
  override protected def _backPropagateGradient(f: FamilyType#ConnectionType): Tensor = weights.value match {
    case w:FixedLayers1DenseTensor3 => //more efficient because Tensor3 is viewed as a seq of matrices
      val outerProd = f._1.value outer f._2.value
      val outGradient = f._3.objectiveGradient
      val weightGradient = new FixedLayers1DenseTensor3(
        (0 until w.dim3).foldLeft(new Array[Tensor2](w.dim3))((a,k) => {
          a(k) = (outerProd * outGradient(k)).asInstanceOf[Tensor2]
          f._1.incrementObjectiveGradient((w.matrices(k) * f._2.value) * outGradient(k))
          f._2.incrementObjectiveGradient((f._1.value * w.matrices(k)) * outGradient(k))
          a
        }))
      weightGradient
    case w: Tensor3 =>
      val outGradient = f._3.objectiveGradient
      val weightGradient = w.blankCopy
      for (k <- 0 until weightGradient.dim3) {
        val outputGradient = outGradient(k)
        for (i <- 0 until weightGradient.dim1)
          for (j <- 0 until weightGradient.dim2) {
            val v1 = f._1.value(i)
            val v2 = f._2.value(j)
            weightGradient.update(i, j, k, v1 * v2 * outputGradient)
            f._1.objectiveGradient.+=(i, v2 * w(i, j, k) * outputGradient)
            f._2.objectiveGradient.+=(j, v1 * w(i, j, k) * outputGradient)
          }
      }
      weightGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor3!")
  }

  override protected def _backPropagate(f: FamilyType#ConnectionType): Unit = {}
}

//Use rather not concatenated version, which is twice as fast
trait ConcatenatedNeuralTensorWeights[N1<:NNLayer,N2<:NNLayer,N3<:NNLayer] extends NNWeights3[N1,N2,N3] {
  override type FamilyType = NeuralTensorWeights[N1,N2,N3]
  override type ConnectionType = Connection
  //dimensionality: (dim1+dim2) x (dim1+dim2) x dim3
  override def weights: Weights3
  override protected def _outputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._3)
  override protected def _inputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._1, f._2)
  override protected def _forwardPropagate(f: FamilyType#ConnectionType): Unit = weights.value match {
    case w:FixedLayers1DenseTensor3 =>
      val concatVector = NNUtils.concatenateTensor1(f._1.value, f._2.value)
      val input = NNUtils.fillDense(f._3.value.length)(k => (w.matrices(k) * concatVector) dot concatVector)
      f._3.incrementInput(input)
    case w: Tensor3 =>
      val input = NNUtils.newDense(f._3.value.length)
      for (k <- 0 until input.dim1)
        for (i <- 0 until f._1.value.dim1 + f._2.value.dim1)
          for (j <- 0 until f._1.value.dim1 + f._2.value.dim1) {
            val v1 = if (i < f._1.value.dim1) f._1.value(i) else f._2.value(i - f._1.value.dim1)
            val v2 = if (j < f._1.value.dim1) f._1.value(j) else f._2.value(j - f._1.value.dim1)
            val weight: Double = weights.value(i, j, k)
            input +=(k, weight * v1 * v2)
          }
      f._3.incrementInput(input)
  }
  override protected def _backPropagateGradient(f: FamilyType#ConnectionType): Tensor = weights.value match {
    case w:FixedLayers1DenseTensor3 =>
      val concatVector = NNUtils.concatenateTensor1(f._1.value, f._2.value)
      val outerProd = concatVector outer concatVector
      val outGradient = f._3.objectiveGradient
      val inputGradient = NNUtils.newDense(f._1.value.dim1 + f._2.value.dim1)
      val weightGradient = new FixedLayers1DenseTensor3(
        (0 until w.dim3).foldLeft(new Array[Tensor2](w.dim3))((a,k) => {
          a(k) = (outerProd * outGradient(k)).asInstanceOf[Tensor2]
          val in_k = w.matrices(k) * concatVector
          in_k += (concatVector * w.matrices(k))
          if(!f._1.isInstanceOf[InputNNLayer] || !f._2.isInstanceOf[InputNNLayer] )
            inputGradient += (in_k, outGradient(k))
          a
        }))
      val (firstGradient,secondGradient) = NNUtils.splitTensor1(inputGradient,f._1.value.dim1)
      f._1.incrementObjectiveGradient(firstGradient)
      f._2.incrementObjectiveGradient(secondGradient)
      weightGradient
    case w: Tensor3 =>
      val outGradient = f._3.objectiveGradient
      val weightGradient = w.blankCopy
      for (k <- 0 until weightGradient.dim3) {
        val outputGradient = outGradient(k)
        for (i <- 0 until weightGradient.dim1)
          for (j <- 0 until weightGradient.dim2) {
            val v1 = if (i < f._1.value.dim1) f._1.value(i) else f._2.value(i - f._1.value.dim1)
            val v2 = if (j < f._1.value.dim1) f._1.value(j) else f._2.value(j - f._1.value.dim1)
            weightGradient.update(i, j, k, v1 * v2 * outputGradient)
            if (i < f._1.value.dim1) f._1.objectiveGradient.+=(i, v2 * w(i, j, k) * outputGradient) else f._2.objectiveGradient.+=(i - f._1.value.dim1, v2 * w(i, j, k) * outputGradient)
            if (j < f._1.value.dim1) f._1.objectiveGradient.+=(j, v1 * w(i, j, k) * outputGradient) else f._2.objectiveGradient.+=(j - f._1.value.dim1, v1 * w(i, j, k) * outputGradient)
          }
      }
      weightGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor3!")
  }

  override protected def _backPropagate(f: FamilyType#ConnectionType): Unit = {}
}

trait SingleTargetNeuronWeights[N1 <: NNLayer, N2 <:NNLayer] extends NNWeights2[N1,N2] {
  override type FamilyType <: SingleTargetNeuronWeights[N1,N2]
  override type ConnectionType = Connection
  override def weights: Weights1
  def targetIndex:Int
  override protected def _outputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._2)
  override protected def _inputLayers(f: FamilyType#ConnectionType): Seq[NNLayer] = Seq(f._1)
  override protected def _forwardPropagate(f: FamilyType#ConnectionType): Unit = weights.value match {
    case weights: Tensor1 =>
      f._2.incrementInput(new SingletonTensor1(f._2.value.dim1,targetIndex,f._1.value dot weights))
  }
  override protected def _backPropagateGradient(f: FamilyType#ConnectionType): Tensor = weights.value match {
    case w: Tensor1 =>
      val outGradient = f._2.objectiveGradient(targetIndex)
      f._1.incrementObjectiveGradient(w * outGradient)
      f._1.value * outGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw objective because this should not be possible
  }

  override protected def _backPropagate(f: FamilyType#ConnectionType): Unit = {}
}

