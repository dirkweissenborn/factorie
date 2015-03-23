package cc.factorie.nn.weights

import cc.factorie.la._
import cc.factorie.model._
import cc.factorie.nn.{RecurrentNNUnit, NNUnit}
import cc.factorie.util.SingletonIndexedSeq
import scala.collection.mutable

trait NNConnection {
  type LayerType <: Layer
  type ConnectionType <: NNConnection
  def weights:Weights
  def numUnits:Int
  def numOutputUnits:Int
  def reset(c: ConnectionType#LayerType):Unit = {}
  protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit]
  protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit]
  //increments input of output layers (by calling l.incrementInput(inc:Tensor1) for each output layer l)
  protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit
  protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit
  //returns objective on weights and propagates objective to input layers from given objective on output layer
  protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor

  trait Layer {
    var disabled:Boolean = false
    //Can be used to store information
    private lazy val featureMap = mutable.Map[String,Any]()
    def setFeature[T](k:String,v:T) = featureMap += k -> v
    def getFeature[T](k:String) = featureMap.get(k).asInstanceOf[Option[T]]
    protected[weights] var _tempInputs:Iterable[Tensor1] = null
    protected[weights] var _tempGradients:Iterable[Tensor1] = null
    protected[weights] var _tempWeightGradient:Tensor = null
    //can be used to not always create new tensor objects on forward pass
    def tempInputs:Iterable[Tensor1] = {
      if(_tempInputs == null)
        _tempInputs = outputUnits.map(l =>
          if(l.input == null) null else l.input.blankCopy)
      _tempInputs.withFilter(_ != null).foreach(_.zero())
      _tempInputs
    }
    //can be used to not always create new tensor objects on backprop
    def tempGradients:Iterable[Tensor1] = {
      if(_tempGradients == null)
        _tempGradients = inputUnits.map(l =>
          if(l.objectiveGradient == null) null else l.objectiveGradient.blankCopy)
      _tempGradients.withFilter(_ != null).foreach(_.zero())
      _tempGradients
    }
    def tempWeightGradient:Tensor = {
      if(_tempWeightGradient == null && weights != null)
        _tempWeightGradient = weights.blankCopy
      if(_tempWeightGradient != null)
        _tempWeightGradient.zero()
      _tempWeightGradient
    }
    def weights = connectionType.weights.value
    def connectionType:ConnectionType = NNConnection.this.asInstanceOf[ConnectionType]
    val outputUnits = NNConnection.this._outputUnits(this.asInstanceOf[ConnectionType#LayerType])
    val inputUnits = NNConnection.this._inputUnits(this.asInstanceOf[ConnectionType#LayerType])
    def forwardPropagate() = NNConnection.this._forwardPropagate(this.asInstanceOf[ConnectionType#LayerType])
    def backPropagateGradient = NNConnection.this._backPropagateGradient(this.asInstanceOf[ConnectionType#LayerType])
    def backPropagate() = NNConnection.this._backPropagate(this.asInstanceOf[ConnectionType#LayerType])
    def reset() = NNConnection.this.reset(this.asInstanceOf[ConnectionType#LayerType])
    def numUnits = NNConnection.this.numUnits
  }
  def newConnection(layers:Seq[NNUnit]):ConnectionType#LayerType
}

trait NNConnection1[N1<:NNUnit] extends NNConnection {
  type LayerType <: Layer
  case class Layer(_1:N1) extends super.Layer
  override val numUnits: Int = 1
  def newConnection(_1:N1):Layer = new Layer(_1)
  override def newConnection(layers: Seq[NNUnit]): ConnectionType#LayerType = {
    assert(layers.size == numUnits, s"Cannot create connection between ${layers.size} layers using ${getClass.getName}")
    newConnection(layers.head.asInstanceOf[N1]).asInstanceOf[ConnectionType#LayerType]
  }
}
trait NNConnection2[N1<:NNUnit,N2<:NNUnit] extends NNConnection  {
  type LayerType <: Layer
  case class Layer(_1:N1,_2:N2) extends super.Layer
  override val numUnits: Int = 2
  def newConnection(_1:N1,_2:N2) = new Layer(_1,_2)
  override def newConnection(layers: Seq[NNUnit]): ConnectionType#LayerType = {
    assert(layers.size == numUnits, s"Cannot create connection between ${layers.size} layers using ${getClass.getName}")
    newConnection(layers.head.asInstanceOf[N1],layers.last.asInstanceOf[N2]).asInstanceOf[ConnectionType#LayerType]
  }
}
trait NNConnection3[N1<:NNUnit,N2<:NNUnit,N3<:NNUnit] extends NNConnection  {
  type LayerType <: Layer
  type ConnectionType <: NNConnection3[N1,N2,N3]
  case class Layer(_1:N1, _2:N2, _3:N3) extends super.Layer
  override val numUnits: Int = 3
  def newConnection(_1:N1,_2:N2,_3:N3) = new Layer(_1,_2,_3)
  override def newConnection(layers: Seq[NNUnit]): ConnectionType#LayerType = {
    assert(layers.size == numUnits, s"Cannot create connection between ${layers.size} layers using ${getClass.getName}")
    newConnection(layers.head.asInstanceOf[N1],layers(1).asInstanceOf[N2],
      layers.last.asInstanceOf[N3]).asInstanceOf[ConnectionType#LayerType]
  }
}
trait NNConnection4[N1<:NNUnit,N2<:NNUnit,N3<:NNUnit,N4<:NNUnit] extends NNConnection {
  type LayerType <: Layer
  case class Layer(_1:N1, _2:N2, _3:N3, _4:N4) extends super.Layer
  override val numUnits: Int = 4
  def newConnection(_1:N1,_2:N2,_3:N3,_4:N4) = new Layer(_1,_2,_3,_4)
  override def newConnection(layers: Seq[NNUnit]): ConnectionType#LayerType = {
    assert(layers.size == numUnits, s"Cannot create connection between ${layers.size} layers using ${getClass.getName}")
    newConnection(layers.head.asInstanceOf[N1],layers(1).asInstanceOf[N2],
      layers(2).asInstanceOf[N3],layers.last.asInstanceOf[N4]).asInstanceOf[ConnectionType#LayerType]
  }
}

trait NNConnectionNTo1[N1<:NNUnit,N2<:NNUnit] extends NNConnection {
  type LayerType <: Layer
  
  override final val numOutputUnits: Int = 1
  override final protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] =
    SingletonIndexedSeq(c.asInstanceOf[Layer]._2)
  override final protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] =
    c.asInstanceOf[Layer]._1

  case class Layer(_1:Seq[N1], _2:N2) extends super.Layer {
    override def numUnits: Int = 1+_1.size
  }
  override val numUnits: Int = -1
  def newConnection(_1:Seq[N1],_2:N2) = new Layer(_1,_2)
  override def newConnection(layers: Seq[NNUnit]): ConnectionType#LayerType = {
    newConnection(layers.tail.asInstanceOf[Seq[N1]],layers.head.asInstanceOf[N2]).asInstanceOf[ConnectionType#LayerType]
  }
}

trait WrappedConnection extends NNConnection {
  protected val underlying:NNConnection
  implicit def toUnderlyingConnection(c:ConnectionType#LayerType) = c.asInstanceOf[underlying.ConnectionType#LayerType]
  def numUnits: Int = underlying.numUnits
  protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] =
    underlying._inputUnits(c)
  protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] =
    underlying._outputUnits(c)
  def numOutputUnits: Int = underlying.numOutputUnits
  def weights: Weights = underlying.weights
  def newConnection(layers: Seq[NNUnit]) = {
    underlying.newConnection(layers).asInstanceOf[ConnectionType#LayerType]
  }
  protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = underlying._backPropagate(c)
  protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = {
    underlying._backPropagateGradient(c)
  }
  protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = {
    underlying._forwardPropagate(c)
  }
}

trait NNConnection1ToN[N1<:NNUnit,N2<:NNUnit] extends NNConnection {
  type LayerType <: Layer

  override val numUnits: Int = -1
  override final val numOutputUnits: Int = Int.MaxValue
  override final protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] =
    c.asInstanceOf[Layer]._2
  override final protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] =
    SingletonIndexedSeq(c.asInstanceOf[Layer]._1)

  case class Layer(_1:N1, _2:Seq[N2]) extends super.Layer {
    override def numUnits: Int = 1+_2.size
  }
  def newConnection(_1:N1,_2:Seq[N2]) = new Layer(_1,_2)
  override def newConnection(layers: Seq[NNUnit]): ConnectionType#LayerType = {
    newConnection(layers.head.asInstanceOf[N1],layers.tail.asInstanceOf[Seq[N2]]).asInstanceOf[ConnectionType#LayerType]
  }
}

//The following classes can be sub-classed in your neural network model
trait Bias[N1<:NNUnit] extends NNConnection1[N1] {
  override type ConnectionType <: Bias[N1]
  override type LayerType = Layer
  override def weights: Weights1
  override def numOutputUnits: Int = 1
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq[NNUnit]()
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor =
    c._1.objectiveGradient
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit =
    c._1.incrementInput(weights.value)
  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}
//_1 is input and _2 is output of these weights, if these weights are used as NeuralNetworkWeights
trait FullConnection[N1 <: NNUnit, N2 <:NNUnit] extends NNConnection2[N1,N2] {
  override type ConnectionType <: FullConnection[N1,N2]
  override type LayerType = Layer
  override def weights: Weights2
  override def numOutputUnits: Int = 1
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._2)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = weights.value match {
    case weights: Tensor2 => 
      val in = c.tempInputs.head
      if(in != null) {
        c._1.value.*(weights, in)
        c._2.incrementInput(in)
      }
  }
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = weights.value match {
    case w: Tensor2 =>
      val outGradient = c._2.objectiveGradient
      val grad = c.tempGradients.head
      if(grad != null) {
        w.*(outGradient,grad)
        c._1.incrementObjectiveGradient(grad)
      }
      (c._1.value outer outGradient).asInstanceOf[Tensor2]
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw objective because this should not be possible
  }

  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}
//This can be useful if you do not want to precalculate something as input for different layers
object IdentityConnection extends NNConnection2[NNUnit,NNUnit] {
  override type LayerType = Layer
  override type ConnectionType = NNConnection2[NNUnit,NNUnit]
  override val weights: Weights2 = null //no weights here
  override def numOutputUnits: Int = 1
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._2)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = c._2.incrementInput(c._1.value)
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor2 = {
      val outGradient = c._2.objectiveGradient
      c._1.incrementObjectiveGradient(outGradient)
      null //Hack, just return an empty tensor because we do not want to update these weights
  }

  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}








