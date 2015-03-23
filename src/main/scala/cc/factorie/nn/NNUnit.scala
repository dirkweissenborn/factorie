package cc.factorie.nn

import cc.factorie.la._
import cc.factorie.model.Weights
import cc.factorie.nn.weights.NNConnection
import cc.factorie.optimize.MultivariateOptimizableObjective
import cc.factorie.optimize.OptimizableObjectives.SquaredMultivariate
import cc.factorie.variable.{TargetVar, LabeledMutableVar, MutableTensorVar}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

trait NNUnit extends MutableTensorVar {
  final override type Value = Tensor1
  def activationFunction:ActivationFunction

  def zeroInput():Unit
  def input:Tensor1
  def incrementInput(in:Tensor1):Unit
  def setInput(i:Tensor1):Unit
  def updateActivation():Unit

  def zeroObjectiveGradient():Unit
  def objectiveGradient:Tensor1
  def setObjectiveGradient(g:Tensor1):Unit
  def incrementObjectiveGradient(in:Tensor1):Unit
  def updateObjectiveGradient():Unit
  def gradientNeedsUpdate:Boolean = true
  def setGradientNeedsUpdate(b:Boolean)
  def reset() = { }  //usually nothing to do

  protected[nn] var inConnections = List[NNConnection#Layer]()
  protected[nn] var outConnections = List[NNConnection#Layer]()
  protected[nn] def addInConnection(c:NNConnection#Layer) = inConnections ::= c //We could check if it is really an incoming connection, but we don't for performance
  protected[nn] def addOutConnection(c:NNConnection#Layer) = outConnections ::= c //We could check if it is really an incoming connection, but we don't for performance
}

trait RecurrentNNUnit extends NNUnit {
  //if true tensors are cached and reused after creation
  def cached:Boolean
  //has to be true only during training
  var keepHistory:Boolean = false
  //something bigger than 0, important to know when a recurrent network has got to stop
  def maxTime:Int
  
  private val _inputHistory = mutable.Stack[Tensor1]()
  private val _activationHistory = mutable.Stack[Tensor1]()
  private val _cachedInputs = mutable.Stack[Tensor1]()
  private val _cachedActivations = mutable.Stack[Tensor1]() 

  abstract override def updateActivation(): Unit = {
    //push old activation
    if(keepHistory && _time > 1) {
      _activationHistory.push(value)
      if(_cachedActivations.nonEmpty) {
        set(_cachedActivations.pop())(null)
        value.zero()
      } else
        set(value.blankCopy)(null)
    }
    super.updateActivation()
  }
  def stepForwardInTime() = { //push input but keep activation
    if(time > 0 && (maxTime == -1 || time < maxTime) && keepHistory) {
      _inputHistory.push(input)
      if(_cachedInputs.nonEmpty) {
        setInput(_cachedInputs.pop())
        zeroInput()
      } else
        setInput(input.blankCopy)
    } else input.zero()
    _time += 1
  }
  def stepBackwardInTime() = {
    if(keepHistory && _inputHistory.nonEmpty) {
      if(cached) {
        _cachedInputs.push(input)
        _cachedActivations.push(value)
      }
      setInput(_inputHistory.pop())
      set(_activationHistory.pop())(null)
    }
    _time -= 1
  }
  
  abstract override def reset() = {
    while(time > 0) stepBackwardInTime()
    super.reset()
  }

  //this is needed for incrementing for previous timestep while keeping the current objective gradient intact
  protected var _tempObjectiveGradient: Tensor1 = value.blankCopy
  override def incrementObjectiveGradient(in: Tensor1): Unit = {
    setGradientNeedsUpdate(true)
    _tempObjectiveGradient += in
  }
  //Switch temp gradient with current
  abstract override def updateObjectiveGradient(): Unit = if(gradientNeedsUpdate) {
    val _t2 = objectiveGradient
    _t2.zero()
    setObjectiveGradient(_tempObjectiveGradient)
    _tempObjectiveGradient = _t2
    super.updateObjectiveGradient()
  }
  
  private var _time:Int = 0
  def time = _time
}

trait OutputNNUnit extends NNUnit {
  def objectiveFunction:MultivariateOptimizableObjective[Tensor1]
  def lastObjective: Double
  def setLastObjective(obj: Double):Unit
}

trait LabeledNNUnit extends OutputNNUnit with LabeledMutableVar  {
  override type TargetType = TargetNNUnit
  //should get updated when objectiveGradient is called on such a Layer, used for calculating objective through the error function
  abstract override def updateActivation(): Unit = {
    super.updateActivation()
    if(objectiveFunction != null) {
      val (v, gradient) = objectiveFunction.valueAndGradient(value, target.value)
      setObjectiveGradient(gradient)
      setLastObjective(v)
    }
  }
  final abstract override def zeroObjectiveGradient(): Unit = {}
}

trait TargetNNUnit extends MutableTensorVar with TargetVar {
  override type Value = Tensor1
  override type AimerType = LabeledNNUnit
}

class BasicNNUnit(t:Tensor1, override val activationFunction:ActivationFunction = ActivationFunction.Sigmoid) extends NNUnit {
  set(t)(null)
  def this(numNeurons:Int, activationFunction:ActivationFunction) = this(TensorUtils.newDense(numNeurons),activationFunction)
  protected var _input: Tensor1 = try { t.blankCopy } catch { case _:Throwable => null }
  protected var _objectiveGradient: Tensor1 = try { t.blankCopy } catch { case _:Throwable => null }
  private var _needsUpdate = false
  def zeroInput() = _input.zero()
  def zeroObjectiveGradient() = _objectiveGradient.zero()
  def objectiveGradient = _objectiveGradient
  def input = _input
  def incrementInput(in:Tensor1) =
    _input += in
  def incrementObjectiveGradient(in:Tensor1):Unit = {
    _objectiveGradient += in
    _needsUpdate = true
  }
  override def updateObjectiveGradient(): Unit = {
    if(_needsUpdate) {
      activationFunction.updateObjective(this)
      _needsUpdate = false
    }    
  }
  def updateActivation() = {
    value := _input
    activationFunction(this)
    value
  }
  override def setInput(i: Tensor1): Unit = _input = i
  override def setObjectiveGradient(g:Tensor1):Unit = {
    _needsUpdate = true
    _objectiveGradient = g
  }
  override def gradientNeedsUpdate = _needsUpdate
  override def setGradientNeedsUpdate(b: Boolean): Unit = _needsUpdate = b
}
trait InputNNUnit extends NNUnit {
  override def input: Tensor1 = null
  override def incrementInput(in: Tensor1): Unit = {}
  override def zeroInput(): Unit = {}
  override val activationFunction: ActivationFunction = ActivationFunction.Linear
}

class BasicOutputNNUnit(initialValue:Tensor1,
                        activationFunction:ActivationFunction = ActivationFunction.SoftMax,
                        override val objectiveFunction:MultivariateOptimizableObjective[Tensor1] = new SquaredMultivariate) extends BasicNNUnit(initialValue,activationFunction) with OutputNNUnit {
  //override def incrementObjectiveGradient(in:Tensor1):Unit = throw new IllegalAccessException("You cannot increment
  // the error of an output layer, it is calculated from its target")
  protected var _lastObjective:Double = 0.0
  override def lastObjective: Double = _lastObjective
  override def setLastObjective(obj: Double): Unit = _lastObjective = obj
}

class BasicTargetNNUnit(targetValue:Tensor1, override val aimer:LabeledNNUnit) extends TargetNNUnit {
  set(targetValue)(null)
}


//Use for example for words as inputs, doesn't make too much sense to use anywhere else
class OneHotNNUnit(t:SingletonBinaryTensor1) extends BasicNNUnit(t) with InputNNUnit

trait AccumulatableNNUnit extends NNUnit {
  def weightAndGradient:Iterable[(Weights,Tensor,Double)]
}