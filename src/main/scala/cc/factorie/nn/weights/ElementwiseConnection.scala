package cc.factorie.nn.weights

import cc.factorie._
import cc.factorie.la.SingletonTensor1
import cc.factorie.model._
import cc.factorie.nn._

trait ElementwiseConnection[N1 <: NNUnit, N2 <:NNUnit] extends NNConnection2[N1,N2] {
  override type ConnectionType <: FullConnection[N1,N2]
  override type LayerType = Layer
  override def weights: Weights1
  override def numOutputUnits: Int = 1
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._2)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = weights.value match {
    case weights: Tensor1 =>
      val in = c.tempInputs.head
      if(in != null) {
        in *= c._1.value
        c._2.incrementInput(in)
      }
  }
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor1 = weights.value match {
    case w: Tensor1 =>
      val outGradient = c._2.objectiveGradient
      val grad = c.tempGradients.head
      if(grad != null) {
        grad += w
        grad *= outGradient
        c._1.incrementObjectiveGradient(grad)
      }
      val wGrad = c.tempWeightGradient.asInstanceOf[Tensor1]
      if(wGrad != null) {
        wGrad += w
        wGrad *= outGradient
      }
      wGrad
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw objective because this should not be possible
  }
  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}

//weights should contain a Tensor of dim1=1, best something like SingletonTensor1 --> should be a scalar
trait ScalingConnection[N1 <: NNUnit, N2 <:NNUnit] extends NNConnection2[N1,N2] {
  override type ConnectionType <: FullConnection[N1,N2]
  override type LayerType = Layer
  override def weights: Weights1
  override def numOutputUnits: Int = 1
  def scalar = weights.value(0)
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._2)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = {
    val in = c.tempInputs.head
    if(in != null) {
      in *= scalar
      c._2.incrementInput(in)
    }
  }
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor1 = {
    val outGradient = c._2.objectiveGradient
    val grad = c.tempGradients.head
    if(grad != null) {
      grad += outGradient
      grad *= weights.value(0)
      c._1.incrementObjectiveGradient(grad)
    }
    var wGrad = 0.0
    c._1.value.foreachActiveElement((i,v) => wGrad += outGradient(i)*v)
    new SingletonTensor1(1,1,wGrad)
  }
  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}
