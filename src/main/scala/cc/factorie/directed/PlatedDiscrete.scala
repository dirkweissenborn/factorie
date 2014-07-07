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

/*
trait PlatedDiscreteGeneratingFactor extends DirectedFactor {
  def prValue(s:StatisticsType, value:Int, index:Int): Double
  def prValue(value:Int, index:Int): Double = prValue(statistics, value, index)
}
*/

object PlatedDiscrete extends DirectedFamily2[DiscreteSeqVar,ProportionsVariable] {
  self =>
  //def pr(ds:Seq[DiscreteValue], p:IndexedSeq[Double]): Double = ds.map(dv => p(dv.intValue)).product
  def pr(ds:DiscreteSeqVar#Value, p:Proportions): Double = ds.foldLeft(1.0)((acc, dv) => acc * p(dv.intValue)) //this does not box, because foldLeft over actual objects and not primitive types
  def logpr(ds:DiscreteSeqVar#Value, p:Proportions): Double = ds.foldLeft(0.0)((acc, dv) => acc + math.log(p(dv.intValue)))
  def sampledValue(d:DiscreteDomain, length:Int, p:Proportions)(implicit random: scala.util.Random): DiscreteSeqVar#Value =
    Vector.fill(length)(d.apply(p.sampleIndex)).asInstanceOf[DiscreteSeqVar#Value]
  case class Factor(override val _1:DiscreteSeqVar, override val _2:ProportionsVariable) extends super.Factor(_1, _2) with SeqGeneratingFactor {
    def proportionalForChildIndex(idx: Int) = _2.value.apply(_1.intValue(idx))
    def pr(child:DiscreteSeqVar#Value, p:Proportions): Double = self.pr(child, p)
    //override def logpr(s:Statistics): Double = self.logpr(s._1, s._2)
    override def sampledValue(implicit random: scala.util.Random): DiscreteSeqVar#Value = self.sampledValue(_1.domain.elementDomain, _1.length, _2.value) // Avoid creating a Statistics
    def sampledValue(p:Proportions)(implicit random: scala.util.Random): DiscreteSeqVar#Value = {
      if (_1.length == 0) IndexedSeq[DiscreteValue]().asInstanceOf[DiscreteSeqVar#Value]
      else self.sampledValue(_1.domain.elementDomain, _1.length, p)
    }
    override def updateCollapsedParentsForIdx(weight: Double, idx: Int): Boolean = {
      _2.incrementMasses(_1(idx).intValue, weight)(null); true
    }
    def updateCollapsedParents(index:Int, weight:Double): Boolean = { _2.value.+=(_1(index).intValue, weight); true }
  }
  def newFactor(a:DiscreteSeqVar, b:ProportionsVariable) = Factor(a, b)
}
