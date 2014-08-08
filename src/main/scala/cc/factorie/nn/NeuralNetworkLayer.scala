package cc.factorie.nn

import cc.factorie.la._
import cc.factorie.optimize.MultivariateOptimizableObjective
import cc.factorie.optimize.OptimizableObjectives.SquaredMultivariate
import cc.factorie.variable.{TargetVar, LabeledMutableVar, MutableTensorVar}

trait NeuralNetworkLayer extends MutableTensorVar {
  final override type Value = Tensor1
  def activationFunction:ActivationFunction

  def zeroInput():Unit
  def input():Tensor1
  def incrementInput(in:Tensor1):Unit
  def updateActivation():Unit

  def zeroObjectiveGradient():Unit
  def objectiveGradient():Tensor1
  def incrementObjectiveGradient(in:Tensor1):Unit
  def updateObjectiveGradient():Unit

  final protected def updateObjectiveGradient(_objectiveGradient:Tensor1):Unit = {
    //This is not nice, however I don't know how to integrate softmax with ActivationFunction nicely, without loosing efficiency
    if(activationFunction == ActivationFunction.SoftMax) {
      val temp = _objectiveGradient.copy
      _objectiveGradient.zero()
      temp.foreachActiveElement((i,obj_i) => {
        value.foreachActiveElement((j,softmax_j) => {
          val partialDerivative = {if(i==j) softmax_j-softmax_j*softmax_j else -softmax_j*value(i)}
          _objectiveGradient += (j,obj_i*partialDerivative)
        })
      })
    } else _objectiveGradient *= activationFunction.applyDerivative(input())
  }
}

trait OutputNeuralNetworkLayer extends NeuralNetworkLayer {
  def objectiveFunction:MultivariateOptimizableObjective[Tensor1]
  def lastObjective: Double
}

trait LabeledNeuralNetworkLayer extends NeuralNetworkLayer with LabeledMutableVar with OutputNeuralNetworkLayer {
  override type TargetType = TargetNeuralNetworkLayer

  //should get updated when error is called on such a Layer, used for calculating objective through the error function
  override def objectiveGradient() = {
    val (v,gradient) = objectiveFunction.valueAndGradient(value.copy,target.value)
    _lastObjective = v
    updateObjectiveGradient(gradient)
    gradient
  }

  //only needed for hidden units after accumulating their error from parent factors
  final override def updateObjectiveGradient(): Unit = {}
  def lastObjective: Double = _lastObjective
  protected var _lastObjective = 0.0
  final override def zeroObjectiveGradient(): Unit = {}
}

trait TargetNeuralNetworkLayer extends MutableTensorVar with TargetVar {
  override type Value = Tensor1
  override type AimerType = LabeledNeuralNetworkLayer
}

class BasicNeuralNetworkLayer(t:Tensor1, override val activationFunction:ActivationFunction = ActivationFunction.Sigmoid) extends NeuralNetworkLayer {
  set(t)(null)
  def this(numNeurons:Int, activationFunction:ActivationFunction) = this(NNUtils.newDense(numNeurons),activationFunction)
  protected lazy val _input: Tensor1 = t.blankCopy
  protected lazy val _objectiveGradient: Tensor1 = t.blankCopy
  def zeroInput() = _input.zero()
  def zeroObjectiveGradient() = _objectiveGradient.zero()
  def objectiveGradient() = _objectiveGradient
  def updateObjectiveGradient():Unit = updateObjectiveGradient(_objectiveGradient)
  def input() = _input
  def incrementInput(in:Tensor1) =
    _input += in
  def incrementObjectiveGradient(in:Tensor1) =
    _objectiveGradient += in
  def updateActivation() = {
    value := _input
    activationFunction(value)
    value
  }
}
trait InputNeuralNetworkLayer extends NeuralNetworkLayer {
  //this is not nice, maybe change that later
  override def objectiveGradient(): Tensor1 = null
  override def input(): Tensor1 = null
  override def incrementInput(in: Tensor1): Unit = {}
  override def incrementObjectiveGradient(in: Tensor1): Unit = {}
  override def zeroInput(): Unit = {}
  override def zeroObjectiveGradient(): Unit = {}
  override def updateActivation(): Unit = {}
}

class BasicOutputNeuralNetworkLayer(targetValue:Tensor1,
                                    activationFunction:ActivationFunction = ActivationFunction.SoftMax,
                                    override val objectiveFunction:MultivariateOptimizableObjective[Tensor1] = new SquaredMultivariate) extends BasicNeuralNetworkLayer(targetValue.copy,activationFunction) with LabeledNeuralNetworkLayer {
  override val target = new BasicTargetNeuralNetworkLayer(targetValue,this)
  override def incrementObjectiveGradient(in:Tensor1) = throw new IllegalAccessException("You cannot increment the error of an output layer, it is calculated from its target")
}

class BasicTargetNeuralNetworkLayer(targetValue:Tensor1, override val aimer:LabeledNeuralNetworkLayer) extends TargetNeuralNetworkLayer {
  set(targetValue)(null)
}


//Use for example for words as inputs, doesn't make too much sense to use anywhere else
class OneHotLayer(t:SingletonBinaryTensor1) extends BasicNeuralNetworkLayer(t) with InputNeuralNetworkLayer