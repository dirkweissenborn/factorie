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
  //returns error on weights and propagates error to input layers from given error on output layer
  protected def _backPropagateError(f: FamilyType#FactorType): Tensor

  trait Factor extends cc.factorie.model.Factor {
    def weights = family.weights.value
    def family:FamilyType = NeuralNetworkWeightsFamily.this.asInstanceOf[FamilyType]
    val outputLayers = NeuralNetworkWeightsFamily.this._outputLayers(this.asInstanceOf[FamilyType#FactorType])
    val inputLayers = NeuralNetworkWeightsFamily.this._inputLayers(this.asInstanceOf[FamilyType#FactorType])
    def forwardPropagate() = NeuralNetworkWeightsFamily.this._forwardPropagate(this.asInstanceOf[FamilyType#FactorType])
    def backPropagateError = NeuralNetworkWeightsFamily.this._backPropagateError(this.asInstanceOf[FamilyType#FactorType])
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
  override protected def _backPropagateError(f: FamilyType#FactorType): Tensor = weights.value match {
    case w: Tensor1 =>
      val errorGradient = f._1.error().copy
      val derivative: Tensor1 = f._1.activationFunction.applyDerivative(f._1.input)
      errorGradient *= derivative
      errorGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw error because this should not be possible
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
  override protected def _backPropagateError(f: FamilyType#FactorType): Tensor2 = weights.value match {
    case w: Tensor2 =>
      val outError = f._2.error().copy
      outError *= f._2.activationFunction.applyDerivative(f._2.input())
      f._1.incrementError(w * outError)
      (f._1.value outer outError).asInstanceOf[Tensor2]
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw error because this should not be possible
  }

  override protected def _backPropagate(f: FamilyType#FactorType): Unit = {}
}
//_1 and _2 are input layers and _3 is output layer if these weights are considered NeuralNetworkWeights, (e.g., see RecurrentNeuralTensorNetworks by Socher et al.)
trait NeuralTensorWeightsFamily[N1<:NeuralNetworkLayer,N2<:NeuralNetworkLayer,N3<:NeuralNetworkLayer] extends NeuralNetworkWeightsFamily3[N1,N2,N3] {
  override type FamilyType = NeuralTensorWeightsFamily[N1,N2,N3]
  override type FactorType = Factor
  //dimensionality: (dim1+dim2) x (dim1+dim2) x dim3
  override def weights: Weights3
  override protected def _outputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._3)
  override protected def _inputLayers(f: FamilyType#FactorType): Seq[NeuralNetworkLayer] = Seq(f._1, f._2)
  override def statistics(v1: Tensor1, v2: Tensor1, v3: Tensor1): Tensor = {
    val combinedV1V2 = NNUtils.newDense(v1.dim1 + v2.dim1)
    combinedV1V2 += v1
    val d = v1.dim1
    v2.foreachActiveElement((i, v) => combinedV1V2 +=(d + i, v))
    combinedV1V2 outer combinedV1V2 outer v3
  }
  //TODO: Make efficient
  override protected def _forwardPropagate(f: FamilyType#FactorType): Unit = {
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
  override protected def _backPropagateError(f: FamilyType#FactorType): Tensor = weights.value match {
    //TODO: Make efficient
    case w: Tensor3 =>
      val outError = f._3.error()
      val derivative = f._3.activationFunction.applyDerivative(f._3.input())
      val weightError = w.blankCopy
      for (k <- 0 until weightError.dim3) {
        val outputError = derivative(k) * outError(k)
        for (i <- 0 until weightError.dim1)
          for (j <- 0 until weightError.dim2) {
            val v1 = if (i < f._1.value.dim1) f._1.value(i) else f._2.value(i - f._1.value.dim1)
            val v2 = if (j < f._1.value.dim1) f._1.value(j) else f._2.value(j - f._1.value.dim1)
            weightError.update(i, j, k, v1 * v2 * outputError)
            if (i < f._1.value.dim1) f._1.error().+=(i, v2 * w(i, j, k) * outputError) else f._2.error().+=(i - f._1.value.dim1, v2 * w(i, j, k) * outputError)
            if (j < f._1.value.dim1) f._1.error().+=(j, v1 * w(i, j, k) * outputError) else f._2.error().+=(j - f._1.value.dim1, v1 * w(i, j, k) * outputError)
          }
      }
      weightError
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor3!")
  }

  override protected def _backPropagate(f: FamilyType#FactorType): Unit = {}
}

