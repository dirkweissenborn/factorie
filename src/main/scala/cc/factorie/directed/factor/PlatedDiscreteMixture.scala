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

package cc.factorie.directed.factor

import cc.factorie.variable._

object PlatedDiscreteMixture extends DirectedFamily3[DiscreteSeqVar, Mixture[ProportionsVariable], DiscreteSeqVar] {
  self =>

  def pr(ds: DiscreteSeqVar#Value, mixture: scala.collection.Seq[Proportions], gates: DiscreteSeqVar#Value): Double = ds.zip(gates).foldLeft(1.0)((acc, tuple) => acc * mixture(tuple._2.intValue).apply(tuple._1.intValue))

  // Make product more efficient
  def logpr(ds: DiscreteSeqVar#Value, mixture: scala.collection.Seq[Proportions], gates: DiscreteSeqVar#Value): Double = ds.zip(gates).foldLeft(1.0)((acc, tuple) => acc + math.log(mixture(tuple._2.intValue).apply(tuple._1.intValue)))

  def sampledValue(d: DiscreteDomain, mixture: scala.collection.Seq[Proportions], gates: DiscreteSeqVar#Value)(implicit random: scala.util.Random): DiscreteSeqVar#Value =
    (for (i <- 0 until gates.length) yield d.apply(mixture(gates(i).intValue).sampleIndex)).asInstanceOf[DiscreteSeqVar#Value]

  case class Factor(override val _1: DiscreteSeqVar, override val _2: Mixture[ProportionsVariable], override val _3: DiscreteSeqVar) extends super.Factor(_1, _2, _3) with DiscreteSeqGeneratingFactor {
    // f._3
    def pr(child: DiscreteSeqVar#Value, mixture: scala.collection.Seq[Proportions], zs: DiscreteSeqVar#Value): Double = self.pr(child, mixture, zs)

    override def logpr(child: DiscreteSeqVar#Value, mixture: scala.collection.Seq[Proportions], zs: DiscreteSeqVar#Value): Double = self.logpr(child, mixture, zs)

    def sampledValue(mixture: scala.collection.Seq[Proportions], zs: DiscreteSeqVar#Value)(implicit random: scala.util.Random): DiscreteSeqVar#Value = self.sampledValue(_1.domain.elementDomain, mixture, zs)

    def prChoosing(child: DiscreteSeqVar#Value, mixture: scala.collection.Seq[Proportions], mixtureIndex: Int): Double = throw new Error("Not yet implemented")

    def sampledValueChoosing(mixture: scala.collection.Seq[Proportions], mixtureIndex: Int)(implicit random: scala.util.Random): DiscreteSeqVar#Value = throw new Error("Not yet implemented")

    //def prValue(s:Statistics, value:Int, index:Int): Double = throw new Error("Not yet implemented")
    /** Update sufficient statistics in collapsed parents, using current value of child, with weight.  Return false on failure. */
    override def updateCollapsedParents(weight: Double) = {
      _3.intValues.zip(_1.intValues).foreach {
        case (parent, child) => _2(parent).incrementMasses(child, weight)(null)
      }
      true
    }

    override def updateCollapsedParentsForIdx(weight: Double, idx: Int): Boolean = {
      _2(_3(idx).intValue).incrementMasses(_1(idx).intValue, weight)(null); true
    }

  }

  def newFactor(a: DiscreteSeqVar, b: Mixture[ProportionsVariable], c: DiscreteSeqVar) = Factor(a, b, c)
}