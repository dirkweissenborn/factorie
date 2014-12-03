package cc.factorie.nn

import cc.factorie.la.{Tensor, UniformTensor1, Tensor1}

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
  def apply(l:NNLayer):Unit
  //transformations directly on input
  def inputDerivative(l: NNLayer): Tensor1
  def typ: String
}

case class DropOutActivation(activation:ActivationFunction) extends BaseActivationFunction {
  def apply(l: NNLayer): Unit = {
    val input = l.input
    activation.apply(l)
    l.value.foreachActiveElement((i,v) => if(Random.nextBoolean()) {l.value.update(i, 0.0); input.update(i,0.0)})
  }
  override def inputDerivative(l: NNLayer): Tensor1 = {
    val t = activation.inputDerivative(l)
    l.value.foreachActiveElement((i,v) => if(v == 0.0) t update (i, 0.0))
    t
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
    def apply(l: NNLayer) = stabilizeInput(l.value, 1).exponentiate()

    def inputDerivative(l: NNLayer): Tensor1 = {
      val c = l.input.copy; stabilizeInput(c, 1).exponentiate(); c
    }

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
    def apply(l: NNLayer) = l.value.foreachActiveElement { case (i, value) =>
      if(value > 1) l.value.update(i, 1.0)
      else if(value < -1) l.value.update(i, -1.0)
        else l.value.update(i, cc.factorie.maths.tanh(value))
    }
    def inputDerivative(l: NNLayer): Tensor1 = {
      val derivative = l.input.copy
      derivative.foreachActiveElement { case (i, value) =>
        if (value < -1) derivative.update(i, -1)
        else if (value > 1) derivative.update(i, 1)
        else derivative.update(i, 1 - math.pow(Math.tanh(value), 2))
      }
      derivative
    }
  }

  object Linear extends BaseActivationFunction {
    def apply(l: NNLayer) = {}
    def inputDerivative(l: NNLayer): Tensor1 = new UniformTensor1(l.input.length, 1.0)
  }

  object RectifiedLinear extends BaseActivationFunction {
    def apply(l: NNLayer) = l.value.foreachActiveElement { case (i, value) => l.value.update(i, math.max(0.0, value))}
    def inputDerivative(l: NNLayer): Tensor1 = new UniformTensor1(l.input.length, 1.0)
  }

  object Sigmoid extends BaseActivationFunction {
    def apply(l: NNLayer) = l.value.foreachActiveElement { case (i, value) => l.value.update(i, cc.factorie.maths.sigmoid(value))}
    def inputDerivative(l: NNLayer): Tensor1 = {
      val derivative = l.input.copy
      derivative.foreachActiveElement { case (i, value) =>
        val s = cc.factorie.maths.sigmoid(value)
        derivative.update(i, s * (1.0 - s))
      }
      derivative
    }
  }

  object SoftMax extends BaseActivationFunction {
    def apply(l: NNLayer) = l.value.expNormalize()
    def inputDerivative(l: NNLayer): Tensor1 = {
      throw new NotImplementedError("There is no direct calculation of of the Softmax derivative for each unit in separate")
    }
  }

  object Tanh extends BaseActivationFunction {
    def apply(l: NNLayer) = l.value.foreachActiveElement { case (i, value) => l.value.update(i, cc.factorie.maths.tanh(value))}
    def inputDerivative(l: NNLayer): Tensor1 = {
      val derivative = l.input.copy
      derivative.foreachActiveElement { case (i, value) => derivative.update(i, 1 - math.pow(Math.tanh(value), 2)) }
      derivative
    }
  }

}

