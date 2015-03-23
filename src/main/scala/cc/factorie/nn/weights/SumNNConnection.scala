package cc.factorie.nn.weights

import cc.factorie.la.Tensor
import cc.factorie.model.Weights
import cc.factorie.nn.NNUnit

object SumNNConnection extends NNConnectionNTo1[NNUnit,NNUnit] {
  override final def weights: Weights = null
  override type ConnectionType = NNConnectionNTo1[NNUnit,NNUnit]
  //increments input of output layers (by calling l.incrementInput(inc:Tensor1) for each output layer l)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = {
    c._1.foreach(l => c._2.incrementInput(l.value))    
  }
  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
  //returns objective on weights and propagates objective to input layers from given objective on output layer
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = {
    c._1.foreach(l => l.incrementObjectiveGradient(c._2.objectiveGradient))
    null
  }
}
