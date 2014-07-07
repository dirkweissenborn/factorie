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

object PlatedDiscreteMixture extends DirectedFamily3[DiscreteSeqVar,Mixture[ProportionsVariable],DiscreteSeqVar] {
  self =>
  //type Seq[+A] = scala.collection.Seq[A]
  def pr(ds:DiscreteSeqVar#Value, mixture:scala.collection.Seq[Proportions], gates:DiscreteSeqVar#Value): Double = ds.zip(gates).map(tuple => mixture(tuple._2.intValue).apply(tuple._1.intValue)).product // Make product more efficient
  //def pr(ds:Seq[DiscreteValue], mixture:Seq[DoubleSeq], gates:Seq[DiscreteValue]): Double = ds.zip(gates).map(tuple => mixture(tuple._2.intValue).apply(tuple._1.intValue)).product
  def logpr(ds:DiscreteSeqVar#Value, mixture:scala.collection.Seq[Proportions], gates:DiscreteSeqVar#Value): Double = ds.zip(gates).map(tuple => math.log(mixture(tuple._2.intValue).apply(tuple._1.intValue))).sum  
  //def logpr(ds:Seq[DiscreteValue], mixture:Seq[DoubleSeq], gates:Seq[DiscreteValue]): Double = ds.zip(gates).map(tuple => math.log(mixture(tuple._2.intValue).apply(tuple._1.intValue))).sum  
  def sampledValue(d:DiscreteDomain, mixture:scala.collection.Seq[Proportions], gates:DiscreteSeqVar#Value)(implicit random: scala.util.Random): DiscreteSeqVar#Value =
    (for (i <- 0 until gates.length) yield d.apply(mixture(gates(i).intValue).sampleIndex)).asInstanceOf[DiscreteSeqVar#Value]
  case class Factor(override val _1:DiscreteSeqVar, override val _2:Mixture[ProportionsVariable], override val _3:DiscreteSeqVar) extends super.Factor(_1, _2, _3) with MixtureFactor with SeqGeneratingFactor with SeqAsParentFactor {
    def gate = throw new Error("Not yet implemented. Need to make PlatedGate be a Gate?") // f._3
    def pr(child:DiscreteSeqVar#Value, mixture:scala.collection.Seq[Proportions], zs:DiscreteSeqVar#Value): Double = self.pr(child, mixture, zs)
    override def logpr(child:DiscreteSeqVar#Value, mixture:scala.collection.Seq[Proportions], zs:DiscreteSeqVar#Value): Double = self.logpr(child, mixture, zs)
    def sampledValue(mixture:scala.collection.Seq[Proportions], zs:DiscreteSeqVar#Value)(implicit random: scala.util.Random): DiscreteSeqVar#Value = self.sampledValue(_1.domain.elementDomain, mixture, zs).asInstanceOf[DiscreteSeqVar#Value]

    def prChoosing(child:DiscreteSeqVar#Value, mixture:scala.collection.Seq[Proportions], mixtureIndex:Int): Double = throw new Error("Not yet implemented")
    def sampledValueChoosing(mixture:scala.collection.Seq[Proportions], mixtureIndex:Int)(implicit random: scala.util.Random): ChildType#Value = throw new Error("Not yet implemented")
    //def prValue(s:Statistics, value:Int, index:Int): Double = throw new Error("Not yet implemented")
    override def updateCollapsedParents(weight: Double) = {
      _3.intValues.zip(_1.intValues).foreach {
        case (parent, child) => _2(parent).incrementMasses(child, weight)(null)
      }
      true
    }
    override def updateCollapsedParentsForIdx(weight: Double, idx: Int): Boolean = {
      _2(_3(idx).intValue).incrementMasses(_1(idx).intValue, weight)(null); true
    }
    override def updateCollapsedParentsForParentIdx(weight: Double, idx: Int): Boolean = updateCollapsedParentsForIdx(weight, idx)//same as above
    def proportionalForChildIndex(idx: Int) = _2(_3.intValue(idx)).value.apply(_1.intValue(idx))
    //the same as prForIndex here
    def proportionalForParentIndex(idx: Int) = proportionalForChildIndex(idx)
  }
  def newFactor(a:DiscreteSeqVar, b:Mixture[ProportionsVariable], c:DiscreteSeqVar) = Factor(a, b, c)
}
