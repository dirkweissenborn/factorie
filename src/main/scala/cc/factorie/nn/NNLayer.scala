package cc.factorie.nn

import cc.factorie.la._
import cc.factorie.optimize.MultivariateOptimizableObjective
import cc.factorie.optimize.OptimizableObjectives.SquaredMultivariate
import cc.factorie.variable.{TargetVar, LabeledMutableVar, MutableTensorVar}

trait NNLayer extends MutableTensorVar {
  final override type Value = Tensor1
  def activationFunction:ActivationFunction

  def zeroInput():Unit
  def input:Tensor1
  def incrementInput(in:Tensor1):Unit
  def updateActivation():Unit

  def zeroObjectiveGradient():Unit
  def objectiveGradient:Tensor1
  def incrementObjectiveGradient(in:Tensor1):Unit
  def updateObjectiveGradient():Unit

  final protected def updateObjectiveGradient(_objectiveGradient:Tensor1):Unit = {
    //This is not nice, however I don't know how to integrate softmax with ActivationFunction nicely, without loosing efficiency
    if(activationFunction == ActivationFunction.SoftMax) {
      var sum = 0.0
      value.foreachActiveElement((i,softmax_i) => {
        sum += softmax_i * _objectiveGradient(i)
      })
      value.foreachActiveElement((j,softmax_j) => {
        _objectiveGradient.update(j,softmax_j*(_objectiveGradient(j)-sum))
      })
    } else _objectiveGradient *= activationFunction.inputDerivative(this)
  }
  protected[nn] var inConnections = List[NNWeights#Connection]()
  protected[nn] var outConnections = List[NNWeights#Connection]()
  protected[nn] def addInConnection(c:NNWeights#Connection) = inConnections ::= c //We could check if it is really an incoming connection, but we don't for performance
  protected[nn] def addOutConnection(c:NNWeights#Connection) = outConnections ::= c //We could check if it is really an incoming connection, but we don't for performance
}

trait OutputNNLayer extends NNLayer {
  def objectiveFunction:MultivariateOptimizableObjective[Tensor1]
  def lastObjective: Double
  def setLastObjective(obj: Double):Unit
}

trait LabeledNNLayer extends NNLayer with LabeledMutableVar with OutputNNLayer {
  override type TargetType = TargetNNLayer

  //should get updated when error is called on such a Layer, used for calculating objective through the error function
  override def objectiveGradient = {
    val (v,gradient) = objectiveFunction.valueAndGradient(value,target.value)
    setLastObjective(v)
    updateObjectiveGradient(gradient)
    gradient
  }

  //only needed for hidden units after accumulating their error from parent connections
  final override def updateObjectiveGradient(): Unit = {}
  final override def zeroObjectiveGradient(): Unit = {}
}

trait TargetNNLayer extends MutableTensorVar with TargetVar {
  override type Value = Tensor1
  override type AimerType = LabeledNNLayer
}

class BasicNNLayer(t:Tensor1, override val activationFunction:ActivationFunction = ActivationFunction.Sigmoid) extends NNLayer {
  set(t)(null)
  def this(numNeurons:Int, activationFunction:ActivationFunction) = this(NNUtils.newDense(numNeurons),activationFunction)
  protected lazy val _input: Tensor1 = t.blankCopy
  protected lazy val _objectiveGradient: Tensor1 = t.blankCopy
  def zeroInput() = _input.zero()
  def zeroObjectiveGradient() = _objectiveGradient.zero()
  def objectiveGradient = _objectiveGradient
  def updateObjectiveGradient():Unit = updateObjectiveGradient(_objectiveGradient)
  def input = _input
  def incrementInput(in:Tensor1) =
    _input += in
  def incrementObjectiveGradient(in:Tensor1) =
    _objectiveGradient += in
  def updateActivation() = {
    value := _input
    activationFunction(this)
    value
  }
}
trait InputNNLayer extends NNLayer {
  //this is not nice, maybe change that later
  override def objectiveGradient: Tensor1 = null
  override def input: Tensor1 = null
  override def incrementInput(in: Tensor1): Unit = {}
  override def incrementObjectiveGradient(in: Tensor1): Unit = {}
  override def zeroInput(): Unit = {}
  override def zeroObjectiveGradient(): Unit = {}
  override def updateActivation(): Unit = { }
  override val activationFunction: ActivationFunction = ActivationFunction.Linear
  override def updateObjectiveGradient(): Unit = {}
}

class BasicOutputNNLayer(initialValue:Tensor1,
                        activationFunction:ActivationFunction = ActivationFunction.SoftMax,
                        override val objectiveFunction:MultivariateOptimizableObjective[Tensor1] = new SquaredMultivariate) extends BasicNNLayer(initialValue,activationFunction) with OutputNNLayer {
  override def incrementObjectiveGradient(in:Tensor1) = throw new IllegalAccessException("You cannot increment the error of an output layer, it is calculated from its target")
  protected var _lastObjective:Double = 0.0
  override def lastObjective: Double = _lastObjective
  override def setLastObjective(obj: Double): Unit = _lastObjective = obj
}

class BasicTargetNNLayer(targetValue:Tensor1, override val aimer:LabeledNNLayer) extends TargetNNLayer {
  set(targetValue)(null)
}


//Use for example for words as inputs, doesn't make too much sense to use anywhere else
class OneHotNNLayer(t:SingletonBinaryTensor1) extends BasicNNLayer(t) with InputNNLayer