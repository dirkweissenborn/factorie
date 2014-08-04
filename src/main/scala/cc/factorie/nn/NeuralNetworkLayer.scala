package cc.factorie.nn

import cc.factorie.la._
import cc.factorie.optimize.MultivariateOptimizableObjective
import cc.factorie.optimize.OptimizableObjectives.SquaredMultivariate
import cc.factorie.variable.{TargetVar, LabeledMutableVar, MutableTensorVar}

trait NeuralNetworkLayer extends MutableTensorVar {
  final override type Value = Tensor1
  def zeroInput():Unit
  def zeroError():Unit
  def error():Tensor1
  def input():Tensor1
  def activationFunction:ActivationFunction
  def incrementInput(in:Tensor1):Unit
  def incrementError(in:Tensor1):Unit
  def updateActivation():Unit
}

trait LabeledNeuralNetworkLayer extends NeuralNetworkLayer with LabeledMutableVar {
  override type TargetType = TargetNeuralNetworkLayer
  def errorFunction:MultivariateOptimizableObjective[Tensor1]
  //should get updated when error is called on such a Layer, used for calculating objective through the error function
  override def error() = {
    val (v,gradient) = errorFunction.valueAndGradient(value.copy,target.value)
    _lastObjective = v
    gradient
  }
  def lastObjective: Double = _lastObjective
  protected var _lastObjective = 0.0
}

trait TargetNeuralNetworkLayer extends MutableTensorVar with TargetVar {
  override type Value = Tensor1
  override type AimerType = LabeledNeuralNetworkLayer
}

class BasicNeuralNetworkLayer(t:Tensor1, override val activationFunction:ActivationFunction = ActivationFunction.Sigmoid) extends NeuralNetworkLayer {
  set(t)(null)
  def this(numNeurons:Int, activationFunction:ActivationFunction) = this(NNUtils.newDense(numNeurons),activationFunction)
  protected lazy val _input: Tensor1 = t.blankCopy
  protected lazy val _error: Tensor1 = t.blankCopy
  def zeroInput() = _input.zero()
  def zeroError() = _error.zero()
  def error() = _error
  def input() = _input
  def incrementInput(in:Tensor1) =
    _input += in
  def incrementError(in:Tensor1) =
    _error += in
  def updateActivation() = {
    value := _input
    activationFunction(value)
    value
  }
}
trait InputNeuralNetworkLayer extends NeuralNetworkLayer {
  //this is not nice, maybe change that later
  override def error(): Tensor1 = null
  override def input(): Tensor1 = null
  override def incrementInput(in: Tensor1): Unit = {}
  override def incrementError(in: Tensor1): Unit = {}
  override def zeroInput(): Unit = {}
  override def zeroError(): Unit = {}
  override def updateActivation(): Unit = {}
}

class BasicOutputNeuralNetworkLayer(targetValue:Tensor1,
                                    activationFunction:ActivationFunction = ActivationFunction.SoftMax,
                                    override val errorFunction:MultivariateOptimizableObjective[Tensor1] = new SquaredMultivariate) extends BasicNeuralNetworkLayer(targetValue.copy,activationFunction) with LabeledNeuralNetworkLayer {
  override val target = new BasicTargetNeuralNetworkLayer(targetValue,this)
  override def incrementError(in:Tensor1) = throw new IllegalAccessException("You cannot increment the error of an output layer, it is calculated from its target")
}

class BasicTargetNeuralNetworkLayer(targetValue:Tensor1, override val aimer:LabeledNeuralNetworkLayer) extends TargetNeuralNetworkLayer {
  set(targetValue)(null)
}


//Use for example for words as inputs, doesn't make too much sense to use anywhere else
class OneHotLayer(t:SingletonBinaryTensor1) extends BasicNeuralNetworkLayer(t) with InputNeuralNetworkLayer