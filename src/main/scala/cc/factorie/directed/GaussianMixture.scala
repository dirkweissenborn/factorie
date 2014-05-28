/* Copyright (C) 2008-2014 University of Massachusetts Amherst.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://github.com/factorie
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

package cc.factorie.directed

import cc.factorie.variable._

object GaussianMixture extends DirectedFamily4[DoubleVariable,Mixture[_ <: DoubleVariable],Mixture[_ <: DoubleVariable],DiscreteVar] {
  case class Factor(override val _1:DoubleVariable, override val _2:Mixture[_ <: DoubleVariable], override val _3:Mixture[_ <: DoubleVariable], override val _4:DiscreteVar) extends super.Factor(_1, _2, _3, _4) {
    def gate = _4
    override def logpr(child:Double, means:Seq[Double], variances:Seq[Double], z:DiscreteVar#Value) = Gaussian.logpr(child, means(z.intValue), variances(z.intValue))
    def pr(child:Double, means:Seq[Double], variances:Seq[Double], z:DiscreteVar#Value) = Gaussian.pr(child, means(z.intValue), variances(z.intValue))
    def sampledValue(means:Seq[Double], variances:Seq[Double], z:DiscreteVar#Value)(implicit random: scala.util.Random): Double = Gaussian.sampledValue(means(z.intValue), variances(z.intValue))
    def prChoosing(child:Double, means:Seq[Double], variances:Seq[Double], mixtureIndex:Int): Double = Gaussian.pr(child, means(mixtureIndex), variances(mixtureIndex)) 
    def sampledValueChoosing(means:Seq[Double], variances:Seq[Double], mixtureIndex:Int)(implicit random: scala.util.Random): Double = Gaussian.sampledValue(means(mixtureIndex), variances(mixtureIndex))
  }
  def newFactor(a:DoubleVariable, b:Mixture[_ <: DoubleVariable], c:Mixture[_ <: DoubleVariable], d:DiscreteVar) = Factor(a, b, c, d)
  
  // A different version in which all the components share the same variance
  case class FactorSharedVariance(override val _1:DoubleVariable, override val _2:Mixture[_ <: DoubleVariable], override val _3:DoubleVariable, override val _4:DiscreteVar) extends DirectedFactorWithStatistics4[DoubleVariable,Mixture[_ <: DoubleVariable],DoubleVariable,DiscreteVar](_1, _2, _3, _4)  {
    def gate = _4
    override def logpr(child:Double, means:Seq[Double], variance:Double, z:DiscreteVar#Value) = Gaussian.logpr(child, means(z.intValue), variance)
    def pr(child:Double, means:Seq[Double], variance:Double, z:DiscreteVar#Value) = Gaussian.pr(child, means(z.intValue), variance)
    def sampledValue(means:Seq[Double], variance:Double, z:DiscreteVar#Value)(implicit random: scala.util.Random): Double = Gaussian.sampledValue(means(z.intValue), variance)
    def prChoosing(child:Double, means:Seq[Double], variance:Double, mixtureIndex:Int): Double = Gaussian.pr(child, means(mixtureIndex), variance) 
    def sampledValueChoosing(means:Seq[Double], variance:Double, mixtureIndex:Int)(implicit random: scala.util.Random): Double = Gaussian.sampledValue(means(mixtureIndex), variance)
  }
  def newFactor(a:DoubleVariable, b:Mixture[_ <: DoubleVariable], c:DoubleVariable, d:DiscreteVar) = FactorSharedVariance(a, b, c ,d)
  def apply(p1:Mixture[_ <: DoubleVariable], p2:DoubleVariable, p3:DiscreteVar)(implicit random: scala.util.Random) = (c:DoubleVariable) => newFactor(c, p1, p2, p3)
}
