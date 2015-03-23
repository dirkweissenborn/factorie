package cc.factorie.nn.weights

import cc.factorie.la.{Tensor1, Tensor}
import cc.factorie.model.Weights
import cc.factorie.nn.{NNUnit, RecurrentNNUnit}

//Connections from and to RecurrentNNUnits are always also recurrent
//However in some cases we need connections from a recurrent unit only at a specific point in time, e.g., the last
//or there is a static input from a static NNUnit that we should cache

//mixin for connections from recurrent units to other units that should only be active at a specific point
//can be used to get out of recurrent parts of the network
trait SpecificTimeConnection extends NNConnection {
  def time:Int
  //increments input of output layers (by calling l.incrementInput(inc:Tensor1) for each output layer l)
  override abstract protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = {
    if(c.inputUnits.forall {
      case r:RecurrentNNUnit => time < 0 || r.time == time
      case _ => true
    }) 
      super._forwardPropagate(c)
  }
  override abstract protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {
    if(c.inputUnits.forall {
      case r:RecurrentNNUnit => time < 0 || r.time == time
      case _ => true
    }) 
      super._backPropagate(c)
  }
  //returns objective on weights and propagates objective to input layers from given objective on output layer
  override abstract protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = {
    if(c.inputUnits.forall {
      case r:RecurrentNNUnit => r.time == time
      case _ => true
    }) 
      super._backPropagateGradient(c)
    else null
  }
}

trait CachedInputConnection extends NNConnection {
  private final val NEEDS_UPDATE = "needs_update"
  private final val CACHED_TENSORS = "cached_tensors"
  abstract override def reset(c: ConnectionType#LayerType): Unit = {
    c.setFeature(NEEDS_UPDATE, true)
    super.reset(c)
  }
  abstract override def newConnection(layers: Seq[NNUnit]) = {
    val c = super.newConnection(layers)
    assert(c.inputUnits.forall(!_.isInstanceOf[RecurrentNNUnit]) &&
      c.outputUnits.forall(_.isInstanceOf[RecurrentNNUnit]),
      "CachedInputConnection makes only sense with connections from static to recurrent units!")
    c.setFeature(NEEDS_UPDATE, true)
    c.setFeature(CACHED_TENSORS, c.outputUnits.map(l =>
      if(l.input == null) null else l.input.blankCopy))
    c
  }
  abstract override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = {
    //After backprop input unit changes usually, so we need an update
    c.setFeature(NEEDS_UPDATE, true)
    super._backPropagateGradient(c)
  }
  abstract override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = {
    //cache input    
    val cachedInputs = c.getFeature[Iterable[Tensor1]](CACHED_TENSORS).get
    if(c.getFeature[Boolean](NEEDS_UPDATE).get) {
      cachedInputs.zip(c.outputUnits).foreach{ case (cache,u) => cache -= u.input }
      super._forwardPropagate(c)
      cachedInputs.zip(c.outputUnits).foreach{ case (cache,u) => cache += u.input }
      c.setFeature(NEEDS_UPDATE, false)
    } else {
      c.outputUnits.zip(cachedInputs).foreach(p => p._1.incrementInput(p._2))      
    }
  }
} 