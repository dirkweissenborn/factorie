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
  def apply(v:Tensor1):Unit
  //transformations directly on input
  def applyDerivative(input: Tensor1): Tensor1
  def typ: String
}

//Mixin for dropout training
trait DropOutActivation extends ActivationFunction {
  abstract override def apply(v1: Tensor1): Unit = {
    super.apply(v1)
    v1.foreachActiveElement((i,v) => if(Random.nextBoolean())v1.+=(i,-v))
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
  override def toString: String = {
    typ
  }
}

object ActivationFunction {
  object Exp extends BaseActivationFunction {
    def apply(input: Tensor1) = stabilizeInput(input, 1).exponentiate()

    def applyDerivative(input: Tensor1): Tensor1 = {
      val c = input.copy
      apply(c)
      c
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
    def apply(input: Tensor1) = input.foreachActiveElement { case (i, value) =>
      if(value > 1) input.update(i, 1.0)
      else if(value < -1) input.update(i, -1.0)
        else input.update(i, cc.factorie.maths.tanh(value))
    }
    def applyDerivative(input: Tensor1): Tensor1 = {
      val derivative = input.copy
      derivative.foreachActiveElement { case (i, value) =>
        if (value < -1) derivative.update(i, -1)
        else if (value > 1) derivative.update(i, 1)
        else derivative.update(i, 1 - math.pow(Math.tanh(value), 2))
      }
      derivative
    }
  }

  object Linear extends BaseActivationFunction {
    def apply(input: Tensor1) = {}
    def applyDerivative(input: Tensor1): Tensor1 = new UniformTensor1(input.length, 1.0)
  }

  object RectifiedLinear extends BaseActivationFunction {
    def apply(input: Tensor1) = input.foreachActiveElement { case (i, value) => input.update(i, math.max(0.0, value))}
    def applyDerivative(input: Tensor1): Tensor1 = new UniformTensor1(input.length, 1.0)
  }

  object Sigmoid extends BaseActivationFunction {
    def apply(input: Tensor1) = input.foreachActiveElement { case (i, value) => input.update(i, cc.factorie.maths.sigmoid(value))}
    def applyDerivative(input: Tensor1): Tensor1 = {
      val derivative = input.copy
      derivative.foreachActiveElement { case (i, value) =>
        val s = cc.factorie.maths.sigmoid(value)
        derivative.update(i, s * (1.0 - s))
      }
      derivative
    }
  }

  object SoftMax extends BaseActivationFunction {
    def apply(input: Tensor1) = input.expNormalize()
    def applyDerivative(input: Tensor1): Tensor1 = {
      throw new NotImplementedError("There is no direct calculation of of the Softmax derivative for each unit in separate")
    }
  }

  object Tanh extends BaseActivationFunction {
    def apply(input: Tensor1) = input.foreachActiveElement { case (i, value) => input.update(i, cc.factorie.maths.tanh(value))}
    def applyDerivative(input: Tensor1): Tensor1 = {
      val derivative = input.copy
      derivative.foreachActiveElement { case (i, value) => derivative.update(i, 1 - math.pow(Math.tanh(value), 2)) }
      derivative
    }
  }

}

