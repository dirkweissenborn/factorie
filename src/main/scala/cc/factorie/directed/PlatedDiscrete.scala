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

import cc.factorie._
import cc.factorie.util.DoubleSeq
import scala.collection.mutable.ArrayBuffer
import cc.factorie.variable._

/*
trait PlatedDiscreteGeneratingFactor extends DirectedFactor {
  def prValue(s:StatisticsType, value:Int, index:Int): Double
  def prValue(value:Int, index:Int): Double = prValue(statistics, value, index)
}
*/

object PlatedDiscrete extends DirectedFamily2[DiscreteSeqVar,ProportionsVariable] {
  self =>
  def pr(ds:DiscreteSeqVar#Value, p:Proportions): Double = ds.foldLeft(1.0)((acc,dv) => acc * p(dv.intValue))
  def logpr(ds:IndexedSeq[DiscreteValue], p:Proportions): Double = ds.foldLeft(0.0)((acc,dv) => acc + math.log(p(dv.intValue)))
  def sampledValue(d:DiscreteDomain, length:Int, p:Proportions)(implicit random: scala.util.Random) =
    Vector.fill(length)(d.apply(p.sampleIndex))
  case class Factor(override val _1:DiscreteSeqVar, override val _2:ProportionsVariable) extends super.Factor(_1, _2) {
    def pr(child:DiscreteSeqVar#Value, p:Proportions): Double = self.pr(child, p)
    //override def logpr(s:Statistics): Double = self.logpr(s._1, s._2)
    override def sampledValue(implicit random: scala.util.Random): DiscreteSeqVar#Value = self.sampledValue(_1.domain.elementDomain, _1.length, _2.value).asInstanceOf[DiscreteSeqVar#Value] // Avoid creating a Statistics
    def sampledValue(p:Proportions)(implicit random: scala.util.Random) = {
      if (_1.length == 0) IndexedSeq[DiscreteValue]()
      else self.sampledValue(_1.domain.elementDomain, _1.length, p)
    }.asInstanceOf[DiscreteSeqVar#Value]

    override def updateCollapsedParentsForIdx(weight:Double,idx:Int): Boolean = { _2.incrementMasses(_1(idx).intValue, weight)(null); true }
    override def updateCollapsedParents(weight: Double) =  super.updateCollapsedParents(weight)
  }
  def newFactor(a:DiscreteSeqVar, b:ProportionsVariable) = Factor(a, b)
}

/*
object PlatedCategorical extends DirectedFamily2[CategoricalSeqVar[String],ProportionsVariable] {
  self =>
  //def pr(ds:Seq[CategoricalValue], p:IndexedSeq[Double]): Double = ds.map(dv => p(dv.intValue)).product
  def pr(ds:IndexedSeq[CategoricalValue[String]], p:Proportions): Double = ds.map(dv => p(dv.intValue)).product // TODO Make this more efficient; this current boxes
  def logpr(ds:IndexedSeq[CategoricalValue[String]], p:Proportions): Double = ds.map(dv => math.log(p(dv.intValue))).sum // TODO Make this more efficient
  def sampledValue(d:CategoricalDomain[String], length:Int, p:Proportions)(implicit random: scala.util.Random): IndexedSeq[CategoricalValue[String]] =
    Vector.fill(length)(d.apply(p.sampleIndex))
  case class Factor(override val _1:CategoricalSeqVar[String], override val _2:ProportionsVariable) extends super.Factor(_1, _2) {
    def pr(child:IndexedSeq[CategoricalValue[String]], p:Proportions): Double = self.pr(child, p)
    //override def logpr(s:Statistics): Double = self.logpr(s._1, s._2)
    override def sampledValue(implicit random: scala.util.Random): CategoricalSeqVar[String]#Value = self.sampledValue(_1.head.domain, _1.length, _2.value) // Avoid creating a Statistics
    def sampledValue(p:Proportions)(implicit random: scala.util.Random): IndexedSeq[CategoricalValue[String]] = {
      if (_1.length == 0) IndexedSeq[CategoricalValue[String]]()
      else self.sampledValue(_1.head.domain, _1.length, p)
    }
    def updateCollapsedParents(index:Int, weight:Double): Boolean = { _2.value.+=(_1(index).intValue, weight); true }
  }
  def newFactor(a:CategoricalSeqVar[String], b:ProportionsVariable) = Factor(a, b)
} */
