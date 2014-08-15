package cc.factorie.nn

import cc.factorie.la._
import cc.factorie.model._

trait NeuralNetworkWeightsFamily {
  type FactorType <: Factor
  type FamilyType <: NeuralNetworkWeightsFamily
  def weights:Weights

  protected def _outputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer]
  protected def _inputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer]
  //increments input of output layers (by calling l.incrementInput(inc:Tensor1) for each output layer l)
  protected def _forwardPropagate(f: FamilyType#FactorType): Unit
  protected def _backPropagate(f: FamilyType#FactorType): Unit
  //returns objective on weights and propagates objective to input layers from given objective on output layer
  protected def _backPropagateGradient(f: FamilyType#FactorType): Tensor

  trait Factor extends cc.factorie.model.Factor {
    def weights = family.weights.value
    def family:FamilyType = NeuralNetworkWeightsFamily.this.asInstanceOf[FamilyType]
    val outputLayers = NeuralNetworkWeightsFamily.this._outputLayers(this.asInstanceOf[FamilyType#FactorType])
    val inputLayers = NeuralNetworkWeightsFamily.this._inputLayers(this.asInstanceOf[FamilyType#FactorType])
    def forwardPropagate() = NeuralNetworkWeightsFamily.this._forwardPropagate(this.asInstanceOf[FamilyType#FactorType])
    def backPropagateGradient = NeuralNetworkWeightsFamily.this._backPropagateGradient(this.asInstanceOf[FamilyType#FactorType])
    def backPropagate() = NeuralNetworkWeightsFamily.this._backPropagate(this.asInstanceOf[FamilyType#FactorType])
  }
}

trait NeuralNetworkWeightsFamily1[N1<:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily {
  type FactorType <: Factor
  class Factor(override val _1:N1) extends DotFactor1[N1](_1) with super.Factor {
    override def statistics(v1: N1#Value): StatisticsType = v1
  }
}
trait NeuralNetworkWeightsFamily2[N1<:NeuralNetworkLayer,N2<:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily  {
  type FactorType <: Factor
  class Factor(override val _1:N1,override val _2:N2) extends DotFactor2[N1,N2](_1,_2) with super.Factor {
    override def statistics(v1: N1#Value, v2: N2#Value): StatisticsType = v1 outer v2
  }
}
trait NeuralNetworkWeightsFamily3[N1<:NeuralNetworkLayer,N2<:NeuralNetworkLayer,N3<:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily  {
  type FactorType <: Factor
  type FamilyType <: NeuralNetworkWeightsFamily3[N1,N2,N3]
  def statistics(v1: N1#Value, v2: N2#Value, v3: N3#Value):Tensor
  class Factor(override val _1:N1, override val _2:N2, override val _3:N3) extends DotFactor3[N1,N2,N3](_1,_2,_3) with super.Factor {
    override def statistics(v1: N1#Value, v2: N2#Value, v3: N3#Value): Tensor = family.statistics(v1,v2,v3)
  }
}
trait NeuralNetworkWeightsFamily4[N1<:NeuralNetworkLayer,N2<:NeuralNetworkLayer,N3<:NeuralNetworkLayer,N4<:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily {
  type FactorType <: Factor
  class Factor(override val _1:N1, override val _2:N2, override val _3:N3, override val _4:N4) extends DotFactor4[N1,N2,N3,N4](_1,_2,_3,_4) with super.Factor {
    override def statistics(v1: N1#Value, v2: N2#Value, v3: N3#Value, v4: N4#Value): Tensor = v1 outer v2 outer v3 outer v4
  }
}

//The following classes can be sub-classed in your neural network model
trait Bias[N1<:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily1[N1] {
  override type FamilyType <: Bias[N1]
  override type FactorType = Factor
  override def weights: Weights1
  override protected def _outputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._1)
  override protected def _inputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq[NeuralNetworkLayer]()
  override protected def _backPropagateGradient(f: FamilyType#FactorType): Tensor = weights.value match {
    case w: Tensor1 => f._1.objectiveGradient()
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw objective because this should not be possible
  }
  override protected def _forwardPropagate(f: FamilyType#FactorType): Unit = {
    f._1.incrementInput(weights.value)
  }
  override protected def _backPropagate(f: FamilyType#FactorType): Unit = {}
}
//_1 is input and _2 is output of these weights, if these weights are used as NeuralNetworkWeights
trait BasicLayerToLayerWeightsFamily[N1 <: NeuralNetworkLayer, N2 <:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily2[N1,N2] {
  override type FamilyType <: BasicLayerToLayerWeightsFamily[N1,N2]
  override type FactorType = Factor
  override def weights: Weights2
  override protected def _outputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._2)
  override protected def _inputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._1)
  override protected def _forwardPropagate(f: FamilyType#FactorType): Unit = weights.value match {
    case weights: Tensor2 => f._2.incrementInput(f._1.value * weights)
  }
  override protected def _backPropagateGradient(f: FamilyType#FactorType): Tensor2 = weights.value match {
    case w: Tensor2 =>
      val outGradient = f._2.objectiveGradient()
      if(!f._1.isInstanceOf[InputNeuralNetworkLayer])
        f._1.incrementObjectiveGradient(w * outGradient)
      (f._1.value outer outGradient).asInstanceOf[Tensor2]
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw objective because this should not be possible
  }

  override protected def _backPropagate(f: FamilyType#FactorType): Unit = {}
}
//_1 and _2 are input layers and _3 is output layer if these weights are considered NeuralNetworkWeights, (e.g., see RecurrentNeuralTensorNetworks by Socher et al.)
trait NeuralTensorWeightsFamily[N1<:NeuralNetworkLayer,N2<:NeuralNetworkLayer,N3<:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily3[N1,N2,N3] {
  override type FamilyType = NeuralTensorWeightsFamily[N1,N2,N3]
  override type FactorType = Factor
  //dimensionality: dim1 x dim2 x dim3
  override def weights: Weights3
  override protected def _outputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._3)
  override protected def _inputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._1, f._2)
  override def statistics(v1: Tensor1, v2: Tensor1, v3: Tensor1): Tensor = {
    v1 outer v2 outer v3
  }
  override protected def _forwardPropagate(f: FamilyType#FactorType): Unit = weights.value match {
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
  override protected def _backPropagateGradient(f: FamilyType#FactorType): Tensor = weights.value match {
    case w:FixedLayers1DenseTensor3 =>
      val outerProd = f._1.value outer f._2.value
      val outGradient = f._3.objectiveGradient()
      val weightGradient = new FixedLayers1DenseTensor3(
        (0 until w.dim3).foldLeft(new Array[Tensor2](w.dim3))((a,k) => {
          a(k) = (outerProd * outGradient(k)).asInstanceOf[Tensor2]
          if(!f._1.isInstanceOf[InputNeuralNetworkLayer])
            f._1.incrementObjectiveGradient((w.matrices(k) * f._2.value) * outGradient(k))
          if(!f._2.isInstanceOf[InputNeuralNetworkLayer])
            f._2.incrementObjectiveGradient((f._1.value * w.matrices(k)) * outGradient(k))
          a
        }))
      weightGradient
    case w: Tensor3 =>
      val outGradient = f._3.objectiveGradient()
      val weightGradient = w.blankCopy
      for (k <- 0 until weightGradient.dim3) {
        val outputGradient = outGradient(k)
        for (i <- 0 until weightGradient.dim1)
          for (j <- 0 until weightGradient.dim2) {
            val v1 = f._1.value(i)
            val v2 = f._2.value(j)
            weightGradient.update(i, j, k, v1 * v2 * outputGradient)
            if(!f._1.isInstanceOf[InputNeuralNetworkLayer])
              f._1.objectiveGradient().+=(i, v2 * w(i, j, k) * outputGradient)
            if(!f._2.isInstanceOf[InputNeuralNetworkLayer])
              f._2.objectiveGradient().+=(j, v1 * w(i, j, k) * outputGradient)
          }
      }
      weightGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor3!")
  }

  override protected def _backPropagate(f: FamilyType#FactorType): Unit = {}
}

//Use rather not concatenated version, which is twice as fast
trait ConcatenatedNeuralTensorWeightsFamily[N1<:NeuralNetworkLayer,N2<:NeuralNetworkLayer,N3<:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily3[N1,N2,N3] {
  override type FamilyType = NeuralTensorWeightsFamily[N1,N2,N3]
  override type FactorType = Factor
  //dimensionality: (dim1+dim2) x (dim1+dim2) x dim3
  override def weights: Weights3
  override protected def _outputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._3)
  override protected def _inputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._1, f._2)
  override def statistics(v1: Tensor1, v2: Tensor1, v3: Tensor1): Tensor = {
    val concatVector = NNUtils.concatenateTensor1(v1, v2)
    concatVector outer concatVector outer v3
  }
  override protected def _forwardPropagate(f: FamilyType#FactorType): Unit = weights.value match {
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
  override protected def _backPropagateGradient(f: FamilyType#FactorType): Tensor = weights.value match {
    case w:FixedLayers1DenseTensor3 =>
      val concatVector = NNUtils.concatenateTensor1(f._1.value, f._2.value)
      val outerProd = concatVector outer concatVector
      val outGradient = f._3.objectiveGradient()
      val inputGradient = NNUtils.newDense(f._1.value.dim1 + f._2.value.dim1)
      val weightGradient = new FixedLayers1DenseTensor3(
        (0 until w.dim3).foldLeft(new Array[Tensor2](w.dim3))((a,k) => {
          a(k) = (outerProd * outGradient(k)).asInstanceOf[Tensor2]
          val in_k = w.matrices(k) * concatVector
          in_k += (concatVector * w.matrices(k))
          if(!f._1.isInstanceOf[InputNeuralNetworkLayer] || !f._2.isInstanceOf[InputNeuralNetworkLayer] )
            inputGradient += (in_k, outGradient(k))
          a
        }))
      val (firstGradient,secondGradient) = NNUtils.splitTensor1(inputGradient,f._1.value.dim1)
      if(!f._1.isInstanceOf[InputNeuralNetworkLayer])
        f._1.incrementObjectiveGradient(firstGradient)
      if( !f._2.isInstanceOf[InputNeuralNetworkLayer] )
        f._2.incrementObjectiveGradient(secondGradient)
      weightGradient
    case w: Tensor3 =>
      val outGradient = f._3.objectiveGradient()
      val weightGradient = w.blankCopy
      for (k <- 0 until weightGradient.dim3) {
        val outputGradient = outGradient(k)
        for (i <- 0 until weightGradient.dim1)
          for (j <- 0 until weightGradient.dim2) {
            val v1 = if (i < f._1.value.dim1) f._1.value(i) else f._2.value(i - f._1.value.dim1)
            val v2 = if (j < f._1.value.dim1) f._1.value(j) else f._2.value(j - f._1.value.dim1)
            weightGradient.update(i, j, k, v1 * v2 * outputGradient)
            if(!f._1.isInstanceOf[InputNeuralNetworkLayer])
              if (i < f._1.value.dim1) f._1.objectiveGradient().+=(i, v2 * w(i, j, k) * outputGradient) else f._2.objectiveGradient().+=(i - f._1.value.dim1, v2 * w(i, j, k) * outputGradient)
            if( !f._2.isInstanceOf[InputNeuralNetworkLayer] )
              if (j < f._1.value.dim1) f._1.objectiveGradient().+=(j, v1 * w(i, j, k) * outputGradient) else f._2.objectiveGradient().+=(j - f._1.value.dim1, v1 * w(i, j, k) * outputGradient)
          }
      }
      weightGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor3!")
  }

  override protected def _backPropagate(f: FamilyType#FactorType): Unit = {}
}

