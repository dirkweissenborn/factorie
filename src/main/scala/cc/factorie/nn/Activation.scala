package cc.factorie.nn

import cc.factorie.la.{Tensor, UniformTensor1, Tensor1}
import cc.factorie.util.{IntSeq, DoubleSeq}

import scala.collection.mutable
import scala.util.Random

/**
 * Created by diwe01 on 15.07.14.
 */
/**
 * An activation function for a hidden layer for neural networks. Adapted from DeepLearning4j (deeplearning4j.org)
 *
 * @author Dirk Weißenborn
 *
 */
trait ActivationFunction {
  def apply(l:NNUnit):Unit
  //transformations directly on input
  def updateObjective(l: NNUnit): Unit
  def typ: String
}

case class DropOutActivation(activation:ActivationFunction,percentage:Double)(implicit rng:Random) extends BaseActivationFunction {
  def apply(l: NNUnit): Unit = {
    val input = l.input
    activation.apply(l)
    l.value.foreachActiveElement((i,v) => if(rng.nextDouble() < percentage) {l.value.update(i, 0.0); input.update(i,0.0)})
  }
  override def updateObjective(l: NNUnit): Unit = {
    activation.updateObjective(l)
    l.value.foreachActiveElement((i,v) => if(v == 0.0) l.objectiveGradient update (i, 0.0))
  }
}

  /**
 * Base activation function: mainly to give the function a canonical representation
 */
trait BaseActivationFunction extends ActivationFunction {
  def typ: String = getClass.getName
  override def equals(o: Any): Boolean = {
    o.getClass.getName == typ
  }
  override def toString: String = typ
}

object ActivationFunction {
  object Exp extends BaseActivationFunction {
    def apply(l: NNUnit) = stabilizeInput(l.value, 1).exponentiate()

    def updateObjective(l: NNUnit): Unit = {
      val c = l.objectiveGradient
      c.foreachElement((i,v) => c(i) = v * math.exp(l.input(i)))
    }

    //copy from deeplearning4j
    /**
     * Ensures numerical stability.
     * Clips values of input such that
     * exp(k * in) is within single numerical precision
     * @param input the input to trim
     * @param k the k (usually 1)
     * @return the stabilized input
     */
    def stabilizeInput(input: Tensor1, k: Double): Tensor1 = {
      val realMin: Double = 1.1755e-38
      val cutOff: Double = math.log(realMin)
      var i: Int = 0
      while (i < input.length) {
        if (input(i) * k > -cutOff) input.update(i, -cutOff / k)
        else if (input(i) * k < cutOff) input.update(i, cutOff / k)
        i += 1
      }
      input
    }
  }

  object HardTanh extends BaseActivationFunction {
    def apply(l: NNUnit) = l.value.foreachActiveElement { case (i, value) =>
      if(value > 1) l.value.update(i, 1.0)
      else if(value < -1) l.value.update(i, -1.0)
        else l.value.update(i, cc.factorie.maths.tanh(value))
    }
    def updateObjective(l: NNUnit): Unit = {
      val grad = l.objectiveGradient
      l.input.foreachActiveElement { case (i, v) =>
        if (v < -1) grad.update(i, 0.0)
        else if (v > 1) grad.update(i, 0.0)
        else grad.update(i, (1 - math.pow(Math.tanh(v), 2)) * grad(i))
      }
    }
  }

  object Linear extends BaseActivationFunction {
    def apply(l: NNUnit) = {}
    def updateObjective(l: NNUnit): Unit = {}
  }

  object RectifiedLinear extends BaseActivationFunction {
    def apply(l: NNUnit) = l.value.foreachActiveElement { case (i, value) => l.value.update(i, math.max(0.0, value))}
    def updateObjective(l: NNUnit): Unit = {
      val grad = l.objectiveGradient
      l.input.foreachActiveElement((i,v) => if(v <= 0.0) grad(i) = 0.0)
    }
  }

  object Sigmoid extends BaseActivationFunction {
    def apply(l: NNUnit) = l.value.foreachActiveElement { case (i, value) => l.value.update(i, cc.factorie.maths.sigmoid(value))}
    def updateObjective(l: NNUnit): Unit = {
      val grad = l.objectiveGradient
      l.value.foreachActiveElement { case (i, value) =>
        grad.update(i, value * (1.0 - value) * grad(i))
      }
    }
  }

  object SoftMax extends BaseActivationFunction {
    def apply(l: NNUnit) = l.value.expNormalize()
    def updateObjective(l: NNUnit): Unit = {
      var sum = 0.0
      l.value.foreachActiveElement((i,softmax_i) => {
        sum += softmax_i * l.objectiveGradient(i)
      })
      l.value.foreachActiveElement((j,softmax_j) => {
        l.objectiveGradient.update(j,softmax_j*(l.objectiveGradient(j)-sum))
      })
    }
  }

  object Tanh extends BaseActivationFunction {
    def apply(l: NNUnit) = l.value.foreachActiveElement { case (i, value) => l.value.update(i, cc.factorie.maths.tanh(value))}
    def updateObjective(l: NNUnit): Unit = {
      val grad = l.objectiveGradient
      l.input.foreachActiveElement { case (i, value) => grad.update(i, grad(i) * (1 - math.pow(Math.tanh(value), 2))) }
    }
  }

  object Squared extends BaseActivationFunction {
    def apply(l: NNUnit) = l.value.foreachActiveElement { case (i, value) => l.value.update(i, value * value)}
    def updateObjective(l: NNUnit): Unit = {
      val grad = l.objectiveGradient
      grad.foreachElement((i,v) => grad(i) = 2*l.input(i)*v)
    }
  }
}

