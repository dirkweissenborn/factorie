package cc.factorie.nn.weights

import cc.factorie.la._
import cc.factorie.model._
import cc.factorie.nn._

import scala.collection.Iterable

trait SingleTargetConnection extends NNConnection {
   def targetIndex:Int
 }


trait SingleTargetNeuronTensorConnection[N1<:NNUnit,N2<:NNUnit,N3<:NNUnit] extends NNConnection3[N1,N2,N3]  with SingleTargetConnection {
  override type ConnectionType = NeuralTensorConnection[N1,N2,N3]
  override type LayerType = Layer
  //dimensionality: dim1 x dim2 x dim3
  override def weights: Weights2
  override def targetIndex:Int
  override def numOutputUnits: Int = 1
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._3)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1, c._2)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = weights.value match {
    case w: Tensor2 =>
      c._3.incrementInput(new SingletonTensor1(c._3.value.dim1,targetIndex,c._1.value * w dot c._2.value))
  }
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = weights.value match {
    case w:Tensor2 =>
      val outerProd = c._1.value outer c._2.value
      val outGradient = c._3.objectiveGradient(targetIndex)
      val grads = c.tempGradients
      val grad1 = grads.head
      val grad2 = grads.last
      if(grad1 != null) {
        w.*(c._2.value,grad1)
        grad1 *= outGradient
        c._1.incrementObjectiveGradient(grad1)
      }
      if(grad2 != null) {
        w.leftMultiply(c._1.value,grad2)
        grad2 *= outGradient
        c._2.incrementObjectiveGradient(grad2)
      }
      outerProd *= outGradient
      outerProd
  }

  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}


trait SingleTargetNeuronConnection[N1 <: NNUnit, N2 <:NNUnit] extends NNConnection2[N1,N2] with SingleTargetConnection {
  override type ConnectionType <: SingleTargetNeuronConnection[N1,N2]
  override type LayerType = Layer
  override def weights: Weights1
  override def targetIndex:Int
  override def numOutputUnits: Int = 1
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._2)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = weights.value match {
    case weights: Tensor1 =>
      c._2.incrementInput(new SingletonTensor1(c._2.value.dim1,targetIndex,c._1.value dot weights))
  }
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = weights.value match {
    case w: Tensor1 =>
      val outGradient = c._2.objectiveGradient(targetIndex)
      val grad = c.tempGradients.head
      if(grad != null) {
        grad += w
        grad *= outGradient
        c._1.incrementObjectiveGradient(grad)
      }
      val wGrad = c.tempWeightGradient.asInstanceOf[Tensor1]
      wGrad += c._1.value
      wGrad *= outGradient
      wGrad
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor2!") //throw objective because this should not be possible
  }

  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}
