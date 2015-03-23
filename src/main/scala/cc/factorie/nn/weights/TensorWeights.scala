package cc.factorie.nn.weights

import cc.factorie.la.{Tensor2, Tensor, Tensor3, FixedLayers1DenseTensor3}
import cc.factorie.model.Weights3
import cc.factorie.nn.{InputNNUnit, TensorUtils, NNUnit}

//_1 and _2 are input layers and _3 is output layer if these weights are considered NeuralNetworkWeights, (e.g., see RecurrentNeuralTensorNetworks by Socher et al.)
trait NeuralTensorConnection[N1<:NNUnit,N2<:NNUnit,N3<:NNUnit] extends NNConnection3[N1,N2,N3] {
  override type ConnectionType = NeuralTensorConnection[N1,N2,N3]
  override type LayerType = Layer
  //dimensionality: dim1 x dim2 x dim3
  override def weights: Weights3
  override def numOutputUnits: Int = 1
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._3)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1, c._2)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = weights.value match {
    case w:FixedLayers1DenseTensor3[_] =>
      val in = c.tempInputs.head
      if(in != null) {
        val temp = TensorUtils.newDense(c._1.value.length)
        var k = 0
        while (k < in.length) {
          w.matrices(k).*(c._2.value, temp)
          in.update(k, temp dot c._1.value)
          k += 1
        }
        c._3.incrementInput(in)
      }
    case w: Tensor3 =>
      val input = c.tempInputs.head
      for (k <- 0 until input.dim1)
        for (i <- 0 until c._1.value.dim1)
          for (j <- 0 until c._2.value.dim1) {
            val v1 = c._1.value(i)
            val v2 =c._2.value(j)
            val weight: Double = weights.value(i, j, k)
            input +=(k, weight * v1 * v2)
          }
      c._3.incrementInput(input)
  }
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = weights.value match {
    case w:FixedLayers1DenseTensor3[_] => //more efficient because Tensor3 is viewed as a seq of matrices
      val outGradient = c._3.objectiveGradient
      val grad1 = c.tempGradients.head
      val grad2 = c.tempGradients.tail.head
      val weightGradient = new FixedLayers1DenseTensor3[Tensor2](
        (0 until w.dim3).foldLeft(new Array[Tensor2](w.dim3))((a,k) => {
          val outerProd = (c._1.value outer c._2.value).asInstanceOf[Tensor2]
          outerProd *= outGradient(k)
          a(k) = outerProd
          w.matrices(k).*(c._2.value,grad1)
          grad1 *= outGradient(k)
          c._1.incrementObjectiveGradient(grad1)
          w.matrices(k).leftMultiply(c._1.value,grad2)
          grad2 *= outGradient(k)
          c._2.incrementObjectiveGradient(grad2)
          a
        }))
      weightGradient
    case w: Tensor3 =>
      val outGradient = c._3.objectiveGradient
      val weightGradient = c.tempWeightGradient.asInstanceOf[Tensor3]
      for (k <- 0 until weightGradient.dim3) {
        val outputGradient = outGradient(k)
        for (i <- 0 until weightGradient.dim1)
          for (j <- 0 until weightGradient.dim2) {
            val v1 = c._1.value(i)
            val v2 = c._2.value(j)
            weightGradient.update(i, j, k, v1 * v2 * outputGradient)
            c._1.objectiveGradient.+=(i, v2 * w(i, j, k) * outputGradient)
            c._2.objectiveGradient.+=(j, v1 * w(i, j, k) * outputGradient)
          }
      }
      weightGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor3!")
  }

  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}

//Use rather not concatenated version, which is twice as fast
trait ConcatenatedNeuralTensorConnection[N1<:NNUnit,N2<:NNUnit,N3<:NNUnit] extends NNConnection3[N1,N2,N3] {
  override type ConnectionType = ConcatenatedNeuralTensorConnection[N1,N2,N3]
  override type LayerType = Layer
  //dimensionality: (dim1+dim2) x (dim1+dim2) x dim3
  override def weights: Weights3
  override def numOutputUnits: Int = 1
  override protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._3)
  override protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq(c._1, c._2)
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = weights.value match {
    case w:FixedLayers1DenseTensor3[_] =>
      val input = c.tempInputs.head
      if(input != null) {
        val concatVector = TensorUtils.concatenateTensor1(c._1.value, c._2.value)
        val temp = TensorUtils.newDense(c._1.value.dim1+ c._2.value.dim1)
        var k = 0
        while (k < input.length) {
          w.matrices(k).*(concatVector, temp)
          input.update(k, temp dot concatVector)
          k += 1
        }
        c._3.incrementInput(input)
      }
    case w: Tensor3 =>
      val input = c.tempInputs.head
      if(input != null) {
        for (k <- 0 until input.dim1)
          for (i <- 0 until c._1.value.dim1 + c._2.value.dim1)
            for (j <- 0 until c._1.value.dim1 + c._2.value.dim1) {
              val v1 = if (i < c._1.value.dim1) c._1.value(i) else c._2.value(i - c._1.value.dim1)
              val v2 = if (j < c._1.value.dim1) c._1.value(j) else c._2.value(j - c._1.value.dim1)
              val weight: Double = weights.value(i, j, k)
              input +=(k, weight * v1 * v2)
            }
        c._3.incrementInput(input)
      }
  }
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = weights.value match {
    case w:FixedLayers1DenseTensor3[_] =>
      val concatVector = TensorUtils.concatenateTensor1(c._1.value, c._2.value)
      val outGradient = c._3.objectiveGradient
      val inputGradient = TensorUtils.newDense(c._1.value.dim1 + c._2.value.dim1)
      val weightGradient = new FixedLayers1DenseTensor3[Tensor2](
        (0 until w.dim3).foldLeft(new Array[Tensor2](w.dim3))((a,k) => {
          val outerProd = (concatVector outer concatVector).asInstanceOf[Tensor2]
          outerProd *= outGradient(k)
          a(k) = outerProd
          val in_k = w.matrices(k) * concatVector
          in_k += (concatVector * w.matrices(k))
          inputGradient += (in_k, outGradient(k))
          a
        }))
      val (firstGradient,secondGradient) = TensorUtils.splitTensor1(inputGradient,c._1.value.dim1)
      c._1.incrementObjectiveGradient(firstGradient)
      c._2.incrementObjectiveGradient(secondGradient)
      weightGradient
    case w: Tensor3 =>
      val outGradient = c._3.objectiveGradient
      val weightGradient = c.tempWeightGradient.asInstanceOf[Tensor3]
      for (k <- 0 until weightGradient.dim3) {
        val outputGradient = outGradient(k)
        for (i <- 0 until weightGradient.dim1)
          for (j <- 0 until weightGradient.dim2) {
            val v1 = if (i < c._1.value.dim1) c._1.value(i) else c._2.value(i - c._1.value.dim1)
            val v2 = if (j < c._1.value.dim1) c._1.value(j) else c._2.value(j - c._1.value.dim1)
            weightGradient.update(i, j, k, v1 * v2 * outputGradient)
            if (i < c._1.value.dim1) c._1.objectiveGradient.+=(i, v2 * w(i, j, k) * outputGradient) else c._2.objectiveGradient.+=(i - c._1.value.dim1, v2 * w(i, j, k) * outputGradient)
            if (j < c._1.value.dim1) c._1.objectiveGradient.+=(j, v1 * w(i, j, k) * outputGradient) else c._2.objectiveGradient.+=(j - c._1.value.dim1, v1 * w(i, j, k) * outputGradient)
          }
      }
      weightGradient
    case _ => throw new IllegalArgumentException(s"Weights value of ${getClass.getSimpleName} must be of type Tensor3!")
  }

  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
}