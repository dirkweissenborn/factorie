package cc.factorie

import cc.factorie.la.Tensor1
import cc.factorie.optimize.MultivariateOptimizableObjective

/**
 * Created by diwe01 on 04.12.14.
 */
package object nn {

  implicit def labelToTensorObjective(obj:MultivariateOptimizableObjective[Int]):MultivariateOptimizableObjective[Tensor1] = new MultivariateOptimizableObjective[Tensor1] {
    override def valueAndGradient(prediction: Tensor1, label: Tensor1): (Double, Tensor1) = {
      val l = label.maxIndex
      obj.valueAndGradient(prediction,l)
    }
  }

}
